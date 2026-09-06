"""Monte Carlo estimation from sampled sequences (Section 3.1, Figure 2c).

Given M sequences drawn from the simulator for one input image, these functions
form the three quantities the framework estimates: marginal probabilities,
conditional probabilities, and time-to-event confidence intervals.

Every function takes `samples` of shape (n_batch, n_samples, seq_len) holding
integer class indices, as returned by `Simulator.sample_sequences`.
"""

import torch


def marginal_probabilities(samples, n_classes):
    """Equation (1): P(Y_i = a | X = x), the fraction of samples in state a at entry i.

    Returns (n_batch, n_classes, seq_len).
    """

    # samples: (n_batch, n_samples, seq_len)

    n_samples = samples.size(1)

    counts = torch.stack(
        [(samples == a).sum(dim=1) for a in range(n_classes)], dim=1
    ).float()
    # (n_batch, n_classes, seq_len)

    return counts / n_samples


def conditional_probabilities(samples, n_classes, condition_entry, condition_class):
    """Equation (2): P(Y_i = a | Y_j = b, X = x).

    Restricts to the sampled sequences whose entry j equals b, then takes the
    fraction of those in state a at entry i. Inputs for which no sample satisfies
    the condition get NaN, and are reported in `n_matching` so a caller can drop
    them rather than silently averaging over an undefined estimate.

    Returns (probabilities, n_matching) of shapes
    (n_batch, n_classes, seq_len) and (n_batch,).
    """

    # samples: (n_batch, n_samples, seq_len)

    matches = samples[:, :, condition_entry] == condition_class
    # (n_batch, n_samples)

    n_matching = matches.sum(dim=1)
    # (n_batch,)

    selected = matches.unsqueeze(-1)
    # (n_batch, n_samples, 1), broadcasts over entries

    counts = torch.stack(
        [((samples == a) & selected).sum(dim=1) for a in range(n_classes)], dim=1
    ).float()
    # (n_batch, n_classes, seq_len)

    denominator = n_matching.view(-1, 1, 1).float()
    probabilities = torch.where(denominator > 0, counts / denominator,
                                torch.full_like(counts, float('nan')))

    return probabilities, n_matching


def time_to_event(sequences, event_class):
    """Index of the first entry equal to `event_class`, per sequence.

    This is the T of Section 3: the time until the game ends, or until a subject
    dies. Sequences in which the event never occurs are right-censored and get
    seq_len, the largest value representable at this sequence length.

    Accepts either (n_batch, n_samples, seq_len) sampled sequences or
    (n_batch, seq_len) ground-truth sequences, and returns the matching shape
    with the last axis removed.
    """

    seq_len = sequences.size(-1)

    matches = sequences == event_class
    occurs = matches.any(dim=-1)

    # argmax over a boolean axis returns the first True; where the event never
    # occurs it returns 0, which `torch.where` then replaces with seq_len. Doing
    # it this way keeps an event at index 0 distinct from an event that never
    # happens -- the two are easy to conflate, and they mean opposite things.
    first_index = matches.float().argmax(dim=-1)

    return torch.where(occurs, first_index, torch.full_like(first_index, seq_len))


def confidence_interval(times, alpha=0.9):
    """The interval I_alpha of Section 3.1, from the (1-alpha)/2 and (1+alpha)/2
    percentiles of the sampled times.

    times: (n_batch, n_samples)
    Returns (lower, upper), each (n_batch,).
    """

    times = times.float()

    lower = torch.quantile(times, (1 - alpha) / 2, dim=1)
    upper = torch.quantile(times, (1 + alpha) / 2, dim=1)

    return lower, upper
