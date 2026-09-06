"""Evaluation metrics for sequence-derived uncertainty estimates (Section 4.1, Appendix A).

Probability estimates are scored one entry at a time and then averaged over entries
to give the sequence-level numbers reported in Tables 1 and 2:

    macro AUC   discriminative ability, one-vs-rest, averaged over classes
    ECE         calibration, confidence expected calibration error
    Brier       squared error against the one-hot label, summed over classes
    CE          negative log probability assigned to the realized class

Confidence intervals are scored by coverage probability, relative width, and
relative mean absolute error (Table 3, Appendix A.2).

Arrays are numpy throughout. `labels` is (n_batch, seq_len) of class indices and
`probabilities` is (n_batch, n_classes, seq_len), as produced by
`monte_carlo.marginal_probabilities`.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# Monte Carlo estimates can be exactly zero, so log probabilities are floored.
PROBABILITY_FLOOR = 1e-12


def macro_auc(labels_at_entry, probabilities_at_entry):
    """One-vs-rest AUC averaged over the classes present at this entry.

    AUC is undefined when every sequence takes the same class at an entry, which
    happens routinely deep into a padded sequence. Those entries return NaN and
    are excluded from the sequence-level average, rather than being scored as a
    perfect 1.0.
    """

    # labels_at_entry: (n_batch,)
    # probabilities_at_entry: (n_batch, n_classes)

    present = np.unique(labels_at_entry)

    if len(present) < 2:
        return np.nan

    one_hot = np.stack([labels_at_entry == c for c in present]).astype(int).T
    # (n_batch, n_present)

    return roc_auc_score(one_hot, probabilities_at_entry[:, present],
                         average='macro', multi_class='ovr')


def brier_score(labels_at_entry, probabilities_at_entry):
    """Squared error against the one-hot label, summed over classes and averaged
    over sequences. This is the sum-over-classes convention, so the score ranges
    over [0, 2] rather than [0, 2/n_classes]."""

    # labels_at_entry: (n_batch,)
    # probabilities_at_entry: (n_batch, n_classes)

    n_classes = probabilities_at_entry.shape[1]
    one_hot = np.eye(n_classes)[labels_at_entry]
    # (n_batch, n_classes)

    return np.sum((probabilities_at_entry - one_hot) ** 2, axis=1).mean()


def cross_entropy(labels_at_entry, probabilities_at_entry):
    """Negative log probability assigned to the realized class."""

    # labels_at_entry: (n_batch,)
    # probabilities_at_entry: (n_batch, n_classes)

    realized = probabilities_at_entry[np.arange(len(labels_at_entry)), labels_at_entry]
    # (n_batch,)

    return float(np.mean(-np.log(realized + PROBABILITY_FLOOR)))


def reliability_curve(labels_at_entry, probabilities_at_entry, n_bins=10):
    """Bin the most-confident prediction and return (mean confidence, accuracy,
    count) per bin, using quantile bins so each holds a similar number of
    sequences. These are the points plotted in the reliability diagrams of
    Figures 3 and 7."""

    # labels_at_entry: (n_batch,)
    # probabilities_at_entry: (n_batch, n_classes)

    confidence = np.max(probabilities_at_entry, axis=1)
    predicted = np.argmax(probabilities_at_entry, axis=1)
    correct = (predicted == labels_at_entry).astype(float)
    # each (n_batch,)

    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, n_bins + 1)))

    if len(edges) < 2:
        # Every prediction carries the same confidence: a single bin.
        return (np.array([confidence.mean()]), np.array([correct.mean()]),
                np.array([len(confidence)]))

    # `right=False` puts each value in [edge_k, edge_{k+1}); the final -1 folds
    # the maximum value back into the last real bin.
    bin_index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, len(edges) - 2)

    mean_confidence, accuracy, counts = [], [], []
    for b in range(len(edges) - 1):
        in_bin = bin_index == b
        if not in_bin.any():
            continue
        mean_confidence.append(confidence[in_bin].mean())
        accuracy.append(correct[in_bin].mean())
        counts.append(int(in_bin.sum()))

    return np.array(mean_confidence), np.array(accuracy), np.array(counts)


def expected_calibration_error(labels_at_entry, probabilities_at_entry, n_bins=10):
    """Confidence ECE (Guo et al., 2017): the gap between confidence and accuracy,
    averaged over quantile bins and weighted by how many sequences fall in each."""

    mean_confidence, accuracy, counts = reliability_curve(
        labels_at_entry, probabilities_at_entry, n_bins
    )

    return float(np.sum(counts * np.abs(accuracy - mean_confidence)) / np.sum(counts))


def evaluate_probabilities(labels, probabilities, n_bins=10):
    """Score probability estimates at every entry and average over entries.

    Returns a dict with the four sequence-level numbers of Tables 1 and 2 and the
    matching per-entry arrays, which are what Figures 3 and 6 plot.
    """

    # labels: (n_batch, seq_len)
    # probabilities: (n_batch, n_classes, seq_len)

    seq_len = labels.shape[1]
    per_entry = {'ECE': [], 'AUC': [], 'CE': [], 'BS': []}

    for i in range(seq_len):
        labels_i = labels[:, i]
        probabilities_i = probabilities[:, :, i]

        per_entry['ECE'].append(expected_calibration_error(labels_i, probabilities_i, n_bins))
        per_entry['AUC'].append(macro_auc(labels_i, probabilities_i))
        per_entry['CE'].append(cross_entropy(labels_i, probabilities_i))
        per_entry['BS'].append(brier_score(labels_i, probabilities_i))

    results = {name: np.array(values) for name, values in per_entry.items()}

    # nanmean so that entries with an undefined AUC drop out of the average
    # instead of dragging it toward an arbitrary value.
    for name in list(results):
        results[f'{name}_per_entry'] = results[name]
        results[name] = float(np.nanmean(results[name]))

    return results


def evaluate_intervals(true_times, lower, upper, sampled_times):
    """Score time-to-event intervals (Table 3, Appendix A.2).

    coverage           fraction of true times inside [lower, upper], endpoints included
    relative_width     mean interval width over mean true time      (Equation 5)
    relative_mae       mean |T - mean sampled T| over mean true time (Equation 6)

    All inputs are 1-D of length n_batch, except `sampled_times` which is
    (n_batch, n_samples).
    """

    true_times = np.asarray(true_times, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Times are integer entry indices, so a true time landing exactly on an
    # endpoint is inside the interval; testing it with a strict inequality would
    # systematically under-report coverage.
    covered = (lower <= true_times) & (true_times <= upper)

    predicted = np.asarray(sampled_times, dtype=float).mean(axis=1)
    # (n_batch,)

    return {
        'coverage': float(covered.mean()),
        'relative_width': float(np.mean(upper - lower) / np.mean(true_times)),
        'relative_mae': float(np.mean(np.abs(true_times - predicted)) / np.mean(true_times)),
    }
