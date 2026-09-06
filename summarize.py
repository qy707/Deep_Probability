"""Aggregate finished runs into the paper's tables and figures.

Reads every run under `--runs_dir`, groups them by scenario and regularization,
averages over seeds, and writes:

    marginal_probability.csv / .md     Table 1 (and Table 2, from conditional runs)
    time_to_event.csv / .md            Table 3
    entry_metrics_<scenario>.png       Figures 3 and 6, entry-wise ECE/AUC/BS/CE
    reliability_<scenario>.png         reliability diagrams, Figures 3 and 7
    intervals_<scenario>.png           Figure 4, interval width and coverage

Each run directory is one produced by train.py, holding run_config.json and an
eval/ subdirectory. Results are reported as mean +/- standard error over the seeds
found for a given (scenario, regularization) pair, as in the paper.

    python summarize.py --runs_dir runs --out_dir results
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs import get_config
from metrics import reliability_curve
from monte_carlo import confidence_interval, marginal_probabilities, time_to_event

# Plot order and styling, matching the figures in the paper.
REGULARIZATIONS = ['none', 'time_dependent', 'constant']
STYLE = {
    'none': ('black', 'No regularization'),
    'time_dependent': ('red', 'Time-dependent reg.'),
    'constant': ('blue', 'Constant reg.'),
}


def latest_epoch_file(run_dir, tag):
    """The metrics file for the last evaluated epoch of a run, or None."""

    pattern = os.path.join(run_dir, 'eval', f'{tag}_epoch_*.json')
    files = glob.glob(pattern)

    if not files:
        return None

    return max(files, key=lambda path: int(path.rsplit('_', 1)[1].split('.')[0]))


def pick_evaluation(run_dir, run_config):
    """Choose which evaluation of a run to report, preferring the test split.

    train.py writes metrics_epoch_<n>.json during training, scored on whichever
    split --eval_split names (validation by default). evaluate.py writes
    metrics_marginal.json, scored on the held-out test split. The test evaluation
    is the one to report when it exists; reading the training-time files instead
    would quietly report validation numbers.

    Returns (metrics_path, per_entry_path, split_name, epoch).
    """

    test_metrics = os.path.join(run_dir, 'eval', 'metrics_marginal.json')

    if os.path.exists(test_metrics):
        return (test_metrics,
                os.path.join(run_dir, 'eval', 'per_entry_marginal.npz'),
                'test',
                run_config.get('epochs'))

    metrics_path = latest_epoch_file(run_dir, 'metrics')

    if metrics_path is None:
        return None, None, None, None

    epoch = int(metrics_path.rsplit('_', 1)[1].split('.')[0])

    return (metrics_path,
            os.path.join(run_dir, 'eval', f'per_entry_epoch_{epoch}.npz'),
            run_config.get('eval_split', 'valid'),
            epoch)


def load_runs(runs_dir):
    """Collect every run under `runs_dir`.

    Each entry records the run's config, its metrics at the last evaluated epoch,
    the matching per-entry arrays, that epoch number, and the run directory.
    """

    runs = []

    for config_path in sorted(glob.glob(os.path.join(runs_dir, '*', 'run_config.json'))):
        run_dir = os.path.dirname(config_path)

        with open(config_path) as handle:
            run_config = json.load(handle)

        metrics_path, per_entry_path, split, epoch = pick_evaluation(run_dir, run_config)
        if metrics_path is None:
            print(f'skipping {run_dir}: no evaluation found')
            continue

        with open(metrics_path) as handle:
            metrics = json.load(handle)

        per_entry = dict(np.load(per_entry_path)) if os.path.exists(per_entry_path) else {}

        runs.append({'config': run_config, 'metrics': metrics, 'per_entry': per_entry,
                     'epoch': epoch, 'split': split, 'dir': run_dir})

    return runs


def mean_and_standard_error(values):
    """Mean and standard error over model realizations, as reported in the paper."""

    values = np.asarray(values, dtype=float)
    mean = float(values.mean())

    if len(values) < 2:
        return mean, 0.0

    return mean, float(values.std(ddof=1) / np.sqrt(len(values)))


def build_table(runs, columns, out_dir, name, caption):
    """Write one table of mean +/- standard error, as CSV and as markdown."""

    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        run_config, metrics = run['config'], run['metrics']
        key = (run_config['scenario'], run_config['regularization'])
        grouped[key]['_epoch'].append(run['epoch'])
        grouped[key]['_split'].append(run['split'])
        for column in columns:
            if column in metrics:
                grouped[key][column].append(metrics[column])

    if not grouped:
        return

    csv_rows = ['scenario,regularization,split,epoch,n_seeds,' + ','.join(f'{c},{c}_se' for c in columns)]
    md_rows = ['| Scenario | Regularization | Split | Epoch | ' + ' | '.join(columns) + ' |',
               '|---' * (len(columns) + 4) + '|']

    for scenario in sorted({key[0] for key in grouped}):
        for regularization in REGULARIZATIONS:
            key = (scenario, regularization)
            if key not in grouped:
                continue

            splits = sorted(set(grouped[key]['_split']))
            if len(splits) > 1:
                print(f'  WARNING: {scenario}/{regularization} mixes evaluations on '
                      f'different splits {splits}')
            split_cell = '/'.join(splits)

            epochs = sorted(set(grouped[key]['_epoch']))
            if len(epochs) > 1:
                print(f'  WARNING: {scenario}/{regularization} mixes runs evaluated at '
                      f'different epochs {epochs}; some runs are probably unfinished')
            epoch_cell = str(epochs[0]) if len(epochs) == 1 else '/'.join(map(str, epochs))

            cells, csv_cells = [], []
            n_seeds = 0
            for column in columns:
                values = grouped[key][column]
                if not values:
                    cells.append('--')
                    csv_cells += ['', '']
                    continue
                n_seeds = max(n_seeds, len(values))
                mean, se = mean_and_standard_error(values)
                cells.append(f'{mean:.4f} ± {se:.4f}')
                csv_cells += [f'{mean:.6f}', f'{se:.6f}']

            csv_rows.append(f'{scenario},{regularization},{split_cell},{epoch_cell},{n_seeds},'
                            + ','.join(csv_cells))
            md_rows.append(f'| {scenario} | {regularization} | {split_cell} | {epoch_cell} | '
                           + ' | '.join(cells) + ' |')

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}.csv'), 'w') as handle:
        handle.write('\n'.join(csv_rows) + '\n')
    with open(os.path.join(out_dir, f'{name}.md'), 'w') as handle:
        handle.write(f'### {caption}\n\n' + '\n'.join(md_rows) + '\n')

    print(f'wrote {os.path.join(out_dir, name)}.csv / .md')


def entry_axis(config):
    """Convert entry index to the time axis used in the paper's figures."""

    seq_len = config['seq_len']

    if config['kind'] == 'atari':
        # Appendix B: t = frame_stride / 60 Hz * i seconds.
        return np.arange(seq_len) * config['frame_stride'] / 60.0, 'Time (seconds)'

    return np.arange(seq_len), 'Time (years)'


def average_over_seeds(curves):
    """Average per-entry curves across seeds, ignoring undefined values.

    AUC is undefined at entries where every sequence takes the same class, and
    those come through as NaN. An entry is averaged over the seeds that did define
    it; an entry no seed defined stays NaN, which matplotlib draws as a gap.
    """

    # curves: (n_seeds, seq_len)

    defined = ~np.isnan(curves)
    n_defined = defined.sum(axis=0)
    # (seq_len,)

    totals = np.where(defined, curves, 0.0).sum(axis=0)
    # (seq_len,)

    return np.where(n_defined > 0, totals / np.maximum(n_defined, 1), np.nan)


def plot_entry_metrics(runs, out_dir):
    """Entry-wise ECE, AUC, BS and CE per scenario (Figures 3 and 6)."""

    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run['per_entry']:
            grouped[run['config']['scenario']][run['config']['regularization']].append(run['per_entry'])

    for scenario, by_regularization in grouped.items():
        config = get_config(scenario)
        time, time_label = entry_axis(config)

        figure, axes = plt.subplots(1, 4, figsize=(20, 4))
        figure.suptitle(scenario, fontsize=14)

        for panel, metric in enumerate(['ECE', 'AUC', 'BS', 'CE']):
            for regularization in REGULARIZATIONS:
                curves = [p[f'{metric}_per_entry'] for p in by_regularization.get(regularization, [])
                          if f'{metric}_per_entry' in p]
                if not curves:
                    continue

                colour, label = STYLE[regularization]
                mean_curve = average_over_seeds(np.stack(curves))
                axes[panel].plot(time[:len(mean_curve)], mean_curve, color=colour, label=label)

            axes[panel].set_xlabel(time_label)
            axes[panel].set_ylabel(metric)
            axes[panel].grid(alpha=0.3)
            axes[panel].legend(fontsize=8)

        figure.tight_layout()
        path = os.path.join(out_dir, f'entry_metrics_{scenario}.png')
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f'wrote {path}')


def plot_reliability(runs, out_dir, entries=(0, 1, 2, 5, 10, 20), n_bins=10):
    """Reliability diagrams at a few entries (Figures 3 and 7).

    Needs the saved Monte Carlo samples, so it is skipped for runs that only kept
    their summary metrics.
    """

    grouped = defaultdict(dict)
    for run in runs:
        samples_path = os.path.join(run['dir'], 'eval', 'samples_marginal.pt')
        truth_path = os.path.join(run['dir'], 'eval', 'ground_truth_marginal.pt')
        if os.path.exists(samples_path) and os.path.exists(truth_path):
            grouped[run['config']['scenario']].setdefault(run['config']['regularization'],
                                                          (samples_path, truth_path))

    for scenario, by_regularization in grouped.items():
        config = get_config(scenario)

        loaded = {}
        for regularization, (samples_path, truth_path) in by_regularization.items():
            samples = torch.load(samples_path).long()
            loaded[regularization] = (
                marginal_probabilities(samples, config['n_classes']).numpy(),
                torch.load(truth_path)['labels'].numpy(),
            )

        # A run may have been trained with --seq_len, so read the length off the
        # saved samples rather than trusting the scenario default.
        seq_len = min(probabilities.shape[-1] for probabilities, _ in loaded.values())
        shown = [i for i in entries if i < seq_len]

        figure, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 3.2), squeeze=False)
        figure.suptitle(f'{scenario}: reliability diagrams', fontsize=14)

        for regularization, (probabilities, labels) in loaded.items():
            colour, label = STYLE[regularization]

            for panel, entry in enumerate(shown):
                confidence, accuracy, _ = reliability_curve(
                    labels[:, entry], probabilities[:, :, entry], n_bins
                )
                axes[0][panel].plot(confidence, accuracy, 'o-', color=colour,
                                    markersize=3, label=label)

        for panel, entry in enumerate(shown):
            axes[0][panel].plot([0, 1], [0, 1], color='orange', linewidth=1)
            axes[0][panel].set_title(f'entry {entry + 1}')
            axes[0][panel].set_xlabel('Estimated probability')
            axes[0][panel].set_xlim(0, 1)
            axes[0][panel].set_ylim(0, 1)
        axes[0][0].set_ylabel('Empirical probability')
        axes[0][-1].legend(fontsize=7)

        figure.tight_layout()
        path = os.path.join(out_dir, f'reliability_{scenario}.png')
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f'wrote {path}')


def plot_intervals(runs, out_dir, alpha=0.9):
    """Interval width against true time, and coverage by true time (Figure 4)."""

    grouped = defaultdict(dict)
    for run in runs:
        samples_path = os.path.join(run['dir'], 'eval', 'samples_marginal.pt')
        truth_path = os.path.join(run['dir'], 'eval', 'ground_truth_marginal.pt')
        if os.path.exists(samples_path) and os.path.exists(truth_path):
            grouped[run['config']['scenario']].setdefault(run['config']['regularization'],
                                                          (samples_path, truth_path))

    for scenario, by_regularization in grouped.items():
        config = get_config(scenario)
        end_token = config['n_classes'] - 1
        shown = [r for r in REGULARIZATIONS if r in by_regularization]

        figure, axes = plt.subplots(2, len(shown), figsize=(5 * len(shown), 8), squeeze=False)
        figure.suptitle(f'{scenario}: {int(alpha * 100)}% time-to-event intervals', fontsize=14)

        for column, regularization in enumerate(shown):
            samples_path, truth_path = by_regularization[regularization]
            samples = torch.load(samples_path).long()
            true_times = torch.load(truth_path)['true_times'].float().numpy()

            sampled_times = time_to_event(samples, end_token).float()
            lower, upper = confidence_interval(sampled_times, alpha)
            width = (upper - lower).numpy()
            covered = (lower.numpy() <= true_times) & (true_times <= upper.numpy())

            axes[0][column].hist2d(true_times, width, bins=40, norm=matplotlib.colors.LogNorm())
            axes[0][column].set_title(STYLE[regularization][1])
            axes[0][column].set_xlabel('True time-to-event')
            axes[0][column].set_ylabel('Interval width')

            bins = np.linspace(0, max(true_times.max(), 1), 25)
            axes[1][column].hist([true_times[covered], true_times[~covered]], bins=bins,
                                 stacked=True, label=['Covered', 'Not covered'])
            axes[1][column].set_xlabel('True time-to-event')
            axes[1][column].set_ylabel('Count')
            axes[1][column].legend(fontsize=8)

        figure.tight_layout()
        path = os.path.join(out_dir, f'intervals_{scenario}.png')
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f'wrote {path}')


def main():

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs_dir', required=True, help='directory holding run subdirectories')
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--alpha', default=0.9, type=float)
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f'no runs with evaluations found under {args.runs_dir}')

    print(f'found {len(runs)} runs')
    os.makedirs(args.out_dir, exist_ok=True)

    build_table(runs, ['ECE', 'AUC', 'CE', 'BS'], args.out_dir,
                'probability_estimation', 'Probability estimation (Tables 1 and 2)')
    build_table(runs, ['coverage', 'relative_width', 'relative_mae'], args.out_dir,
                'time_to_event', 'Time-to-event confidence intervals (Table 3)')
    build_table(runs, ['ground_truth_rmse'], args.out_dir,
                'facemed_ground_truth', 'FaceMed RMSE against exact marginals (Section 6.1)')

    if not args.no_figures:
        plot_entry_metrics(runs, args.out_dir)
        plot_reliability(runs, args.out_dir)
        plot_intervals(runs, args.out_dir, alpha=args.alpha)


if __name__ == '__main__':
    main()
