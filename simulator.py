"""The neural network simulator of Section 3.2.

A CNN encodes the input image into a single embedding, which is handed to an
autoregressive decoder as its first input. The decoder then estimates

    p_theta(Y_i = y_i | X = x, Y_1 = y_1, ..., Y_{i-1} = y_{i-1})

one entry at a time. Two decoders are provided: the LSTM used throughout the main
text, and the Transformer of Appendix G. Both expose the same three methods, so
training and evaluation code never needs to know which one it holds:

    forward(images, prefix)        teacher-forced pass used for training
    sample_sequences(...)          draw Monte Carlo sequences (Figure 2b)
    (constructed from embed_size, n_classes, ...)

Shape conventions follow the paper: sequences are (seq_len, n_batch) inside the
decoders and (n_batch, seq_len) everywhere else.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


class EncoderCNN(nn.Module):
    """ResNet-18 image encoder (Appendix C.1)."""

    def __init__(self, embed_size):
        super().__init__()

        self.embed_size = embed_size
        self.cnn = M.resnet18(num_classes=embed_size)

    def forward(self, images):

        # images: (n_batch, 3, height, width)

        features = self.cnn(images)
        # (n_batch, embed_size)

        return features


class DecoderRNN(nn.Module):
    """Single-layer LSTM decoder (Appendix C.1)."""

    def __init__(self, embed_size, hidden_size, n_classes, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.num_layers = num_layers

        self.embed = nn.Embedding(n_classes, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, features, prefix):

        # features: (n_batch, embed_size)
        # prefix: (seq_len - 1, n_batch), the ground-truth entries y_1 .. y_{l-1}

        embeddings = self.embed(prefix)
        # (seq_len - 1, n_batch, embed_size)

        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # (seq_len, n_batch, embed_size); the image stands in for the missing y_0

        hiddens, _ = self.lstm(embeddings)
        # (seq_len, n_batch, hidden_size)

        logits = self.linear(hiddens)
        # (seq_len, n_batch, n_classes)

        return logits

    def start(self, features):
        """Begin a rollout: the image embedding is the first decoder input."""

        # features: (n_batch, embed_size)

        return features.unsqueeze(0), None
        # (1, n_batch, embed_size), no recurrent state yet

    def step(self, decoder_input, state, position):
        """Advance one entry. `position` is unused by the LSTM but kept for a
        common interface with the Transformer, which needs it for its encoding."""

        # decoder_input: (1, n_batch, embed_size)

        hiddens, state = self.lstm(decoder_input, state)
        # (1, n_batch, hidden_size)

        logits = self.linear(hiddens.squeeze(0))
        # (n_batch, n_classes)

        return logits, state

    def embed_entry(self, entry):
        """Turn a sampled entry into the next decoder input."""

        # entry: (n_batch,)

        return self.embed(entry).unsqueeze(0)
        # (1, n_batch, embed_size)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions for the Transformer decoder."""

    def __init__(self, embed_size, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, embed_size)
        # (max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        # (embed_size // 2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x, position=None):

        # x: (seq_len, n_batch, embed_size), or (1, n_batch, embed_size) when
        #    decoding a single entry at index `position`

        if position is not None:
            return x + self.pe[position].unsqueeze(0).unsqueeze(0)
            # (1, 1, embed_size) broadcast over the batch

        return x + self.pe[:x.size(0)].unsqueeze(1)
        # (seq_len, 1, embed_size) broadcast over the batch


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with an optional key/value cache.

    During training the whole sequence is passed at once and a triangular mask
    enforces causality. During sampling one entry is passed at a time and the
    cached keys and values already contain only past entries, so no mask is needed.
    """

    def __init__(self, embed_size, n_heads, dropout=0.0):
        super().__init__()

        assert embed_size % n_heads == 0, 'embed_size must be divisible by n_heads'

        self.n_heads = n_heads
        self.d_k = embed_size // n_heads

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """Fan the embedding dimension out into separate attention heads."""

        # x: (n_batch, seq_len, embed_size)

        n_batch, seq_len, _ = x.size()

        return x.view(n_batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # (n_batch, n_heads, seq_len, d_k)

    def combine_heads(self, x):
        """Fold the attention heads back into a single embedding dimension."""

        # x: (n_batch, n_heads, seq_len, d_k)

        n_batch, n_heads, seq_len, d_k = x.size()

        return x.transpose(1, 2).contiguous().view(n_batch, seq_len, n_heads * d_k)
        # (n_batch, seq_len, embed_size)

    def forward(self, x, past_kv=None):

        # x: (n_batch, seq_len, embed_size)
        # past_kv: pair of (n_batch, n_heads, past_len, d_k) tensors, or None

        seq_len = x.size(1)

        queries = self.split_heads(self.q_proj(x))
        keys = self.split_heads(self.k_proj(x))
        values = self.split_heads(self.v_proj(x))
        # each (n_batch, n_heads, seq_len, d_k)

        if past_kv is not None:
            keys = torch.cat([past_kv[0], keys], dim=2)
            values = torch.cat([past_kv[1], values], dim=2)
            # (n_batch, n_heads, past_len + seq_len, d_k)

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.d_k)
        # (n_batch, n_heads, seq_len, total_len)

        if past_kv is None:
            causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        attention = self.dropout(torch.softmax(scores, dim=-1))
        # (n_batch, n_heads, seq_len, total_len)

        output = self.combine_heads(attention @ values)
        # (n_batch, seq_len, embed_size)

        return self.out_proj(output), (keys, values)


class TransformerDecoderBlock(nn.Module):
    """One post-norm decoder block: causal attention then a feed-forward layer."""

    def __init__(self, embed_size, n_heads, dropout=0.0):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(embed_size, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):

        # x: (n_batch, seq_len, embed_size)

        attended, new_kv = self.self_attention(x, past_kv)
        x = self.norm1(x + self.dropout(attended))
        # (n_batch, seq_len, embed_size)

        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        # (n_batch, seq_len, embed_size)

        return x, new_kv


class DecoderTransformer(nn.Module):
    """Transformer decoder used for the Appendix G experiments."""

    def __init__(self, embed_size, n_heads, n_classes, num_layers=1, max_len=1000):
        super().__init__()

        self.embed_size = embed_size
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.num_layers = num_layers

        self.embed = nn.Embedding(n_classes, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.linear = nn.Linear(embed_size, n_classes)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_size, n_heads) for _ in range(num_layers)]
        )

    def forward(self, features, prefix):

        # features: (n_batch, embed_size)
        # prefix: (seq_len - 1, n_batch)

        embeddings = self.embed(prefix)
        # (seq_len - 1, n_batch, embed_size)

        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # (seq_len, n_batch, embed_size)

        embeddings = self.positional_encoding(embeddings).transpose(0, 1)
        # (n_batch, seq_len, embed_size)

        for layer in self.layers:
            embeddings, _ = layer(embeddings, None)
        # (n_batch, seq_len, embed_size)

        logits = self.linear(embeddings.transpose(0, 1))
        # (seq_len, n_batch, n_classes)

        return logits

    def start(self, features):
        """Begin a rollout with an empty key/value cache, one entry per layer."""

        # features: (n_batch, embed_size)

        return features.unsqueeze(0), [None] * self.num_layers

    def step(self, decoder_input, state, position):
        """Advance one entry, extending each layer's key/value cache.

        Unlike the LSTM the Transformer has no recurrent state, so `position` is
        needed here to place the entry in the positional encoding."""

        # decoder_input: (1, n_batch, embed_size)
        # state: list of per-layer key/value caches

        embeddings = self.positional_encoding(decoder_input, position).transpose(0, 1)
        # (n_batch, 1, embed_size)

        new_state = []
        for layer, past_kv in zip(self.layers, state):
            embeddings, kv = layer(embeddings, past_kv)
            new_state.append(kv)
        # (n_batch, 1, embed_size)

        logits = self.linear(embeddings.squeeze(1))
        # (n_batch, n_classes)

        return logits, new_state

    def embed_entry(self, entry):
        """Turn a sampled entry into the next decoder input."""

        # entry: (n_batch,)

        return self.embed(entry).unsqueeze(0)
        # (1, n_batch, embed_size)


class Simulator(nn.Module):
    """CNN encoder plus an autoregressive decoder: the simulator of Figure 2.

    decoder_type is 'rnn' (main text) or 'transformer' (Appendix G).
    """

    def __init__(self, n_classes, decoder_type='rnn', embed_size=256, hidden_size=256,
                 n_heads=4, num_layers=1):
        super().__init__()

        self.n_classes = n_classes
        self.decoder_type = decoder_type

        self.encoder = EncoderCNN(embed_size)

        if decoder_type == 'rnn':
            self.decoder = DecoderRNN(embed_size, hidden_size, n_classes, num_layers)
        elif decoder_type == 'transformer':
            self.decoder = DecoderTransformer(embed_size, n_heads, n_classes, num_layers)
        else:
            raise ValueError(f"decoder_type must be 'rnn' or 'transformer', got {decoder_type!r}")

    def forward(self, images, prefix):
        """Teacher-forced pass: predict every entry from the image and the true past."""

        # images: (n_batch, 3, height, width)
        # prefix: (seq_len - 1, n_batch)

        features = self.encoder(images)
        # (n_batch, embed_size)

        logits = self.decoder(features, prefix)
        # (seq_len, n_batch, n_classes)

        return logits

    @torch.no_grad()
    def sample_sequences(self, images, seq_len, n_samples, condition_first_entry=None,
                         max_parallel_samples=None):
        """Draw `n_samples` sequences per image by ancestral sampling (Figure 2b).

        The image is encoded once and its embedding is replicated across the
        Monte Carlo draws, so a chunk of draws rolls out as a single batch rather
        than one rollout at a time. `max_parallel_samples` caps how many draws
        share a rollout, to bound memory on long sequences; None means all of them.

        If `condition_first_entry` is given, every sequence is forced to start with
        that entry, which is how the conditional probabilities of Equation (2) are
        estimated (Appendix D.2).

        Returns (n_batch, n_samples, seq_len) of int64 entries.
        """

        n_batch = images.size(0)
        device = images.device
        chunk = max_parallel_samples or n_samples

        # Sampling is inference: force eval mode so the encoder's batch-norm uses
        # its running statistics. Left in train mode it would normalize with the
        # statistics of whichever images happened to share the batch, making one
        # image's Monte Carlo draws depend on its neighbours. The previous mode is
        # restored so this is safe to call mid-training.
        was_training = self.training
        self.eval()

        try:
            features = self.encoder(images)
            # (n_batch, embed_size)

            chunks = []
            drawn = 0

            while drawn < n_samples:
                width = min(chunk, n_samples - drawn)

                # Repeat each image's embedding `width` times, so rows
                # [j * width : (j + 1) * width] all belong to image j.
                repeated = features.repeat_interleave(width, dim=0)
                # (n_batch * width, embed_size)

                chunks.append(self._rollout(repeated, seq_len, condition_first_entry, device)
                              .view(n_batch, width, seq_len))
                drawn += width

            return torch.cat(chunks, dim=1)
            # (n_batch, n_samples, seq_len)

        finally:
            self.train(was_training)

    def _rollout(self, features, seq_len, condition_first_entry, device):
        """Sample one sequence for every row of `features`."""

        # features: (n_rows, embed_size)

        n_rows = features.size(0)
        sequences = torch.zeros((n_rows, seq_len), dtype=torch.long, device=device)

        decoder_input, state = self.decoder.start(features)
        # (1, n_rows, embed_size)

        for i in range(seq_len):
            logits, state = self.decoder.step(decoder_input, state, i)
            # (n_rows, n_classes)

            if i == 0 and condition_first_entry is not None:
                entry = torch.full((n_rows,), condition_first_entry,
                                   dtype=torch.long, device=device)
            else:
                entry = torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(-1)
            # (n_rows,)

            sequences[:, i] = entry
            decoder_input = self.decoder.embed_entry(entry)
            # (1, n_rows, embed_size)

        return sequences
        # (n_rows, seq_len)
