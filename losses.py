"""The training loss of Equation (4): cross entropy plus a time-dependent logit penalty.

    L(theta) = E_(x,y) [ sum_i  -log p_theta(y_i | x, y_1..y_{i-1})  +  lambda_i * ||z_i||^2 ]

where z_i is the logit vector produced when predicting entry i. Setting every
lambda_i to zero recovers the unregularized maximum-likelihood baseline of
Section 5, so the three variants compared in the paper -- no regularization,
time-dependent, and constant -- are all this one loss with a different lambda
vector (see `configs.lambda_schedule`).

Note on the exponent: Equation (4) in the paper is written with ||z||_2, while
the experiments penalize the *squared* norm ||z||_2^2. The squared form is what
corresponds to the zero-mean Gaussian prior over logits described in Section 6,
and is what this implementation uses. Pass squared=False for the literal
Equation (4) form.
"""

import torch
import torch.nn as nn


class SequenceLoss(nn.Module):
    """Per-entry cross entropy with a per-entry logit-norm penalty.

    The loss is summed over the entries of a sequence and averaged over the batch,
    so its value is comparable across batch sizes and reads as "nats per sequence".
    """

    def __init__(self, lambdas, squared=True):
        super().__init__()

        # lambdas: sequence of seq_len floats, the lambda_i of Equation (4)
        self.register_buffer('lambdas', torch.as_tensor(lambdas, dtype=torch.float))
        self.squared = squared
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):

        # logits: (n_batch, n_classes, seq_len)
        # targets: (n_batch, seq_len)

        seq_len = logits.size(2)
        if self.lambdas.numel() != seq_len:
            raise ValueError(
                f'lambda schedule has {self.lambdas.numel()} entries '
                f'but the sequence has {seq_len}'
            )

        per_entry_ce = self.cross_entropy(logits, targets)
        # (n_batch, seq_len)

        logit_norm = torch.linalg.vector_norm(logits, dim=1)
        # (n_batch, seq_len), the norm ||z_i||_2 of each entry's logit vector

        penalty = logit_norm ** 2 if self.squared else logit_norm
        # (n_batch, seq_len)

        per_entry_loss = per_entry_ce + self.lambdas * penalty
        # (n_batch, seq_len); lambdas broadcasts along the batch

        return per_entry_loss.sum(dim=1).mean()
        # scalar: summed over entries, averaged over the batch
