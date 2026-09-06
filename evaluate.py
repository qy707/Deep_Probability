"""Evaluate a trained simulator with the Monte Carlo framework.

Draws sequences from the simulator for every test image, forms the estimates of
Section 3.1, and scores them with the metrics of Section 4.1. Used both by
train.py during training and as a standalone script on a saved checkpoint:

    python evaluate.py --scenario seaquest --data_dir /data/Atari_HEAD \
        --checkpoint runs/seaquest_time_dependent/checkpoint_epoch_200.pt \
        --out_dir runs/seaquest_time_dependent/eval

Passing --condition_first_entry evaluates the conditional probabilities of
Table 2 instead of the marginal probabilities of Table 1: the test set is
restricted to sequences that start with that entry, and every sampled sequence is
forced to start there too.
"""

import argparse
import json
import os

import numpy as np
import torch

from configs import get_config
from datasets import build_datasets
from data_facemed import ground_truth_marginals
from metrics import evaluate_intervals, evaluate_probabilities
from monte_carlo import confidence_interval, marginal_probabilities, time_to_event
from simulator import Simulator


def collect_samples(model, loader, seq_len, n_samples, device,
                    condition_first_entry=None, max_parallel_samples=None):
    """Run the simulator over a loader and gather its Monte Carlo draws.

    Returns (samples, labels, true_times, ages) where samples is
    (n_data, n_samples, seq_len), labels and true_times are (n_data,...) and ages
    is None for scenarios that do not carry one.
    """

    model.eval()
    samples, labels, true_times, ages = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)

            drawn = model.sample_sequences(
                images, seq_len, n_samples,
                condition_first_entry=condition_first_entry,
                max_parallel_samples=max_parallel_samples,
            )
            # (n_batch, n_samples, seq_len)

            # int8 keeps the sample store small; 19 classes fit comfortably.
            samples.append(drawn.to(torch.int8).cpu())
            labels.append(batch['sequence'])
            true_times.append(batch['length'])
            if 'age' in batch:
                ages.append(batch['age'])

    return (
        torch.cat(samples),
        torch.cat(labels),
        torch.cat(true_times),
        torch.cat(ages) if ages else None,
    )


def evaluate_samples(samples, labels, true_times, config, alpha=0.9, ages=None):
    """Score one set of Monte Carlo draws against the ground truth.

    Returns a dict holding the sequence-level numbers of Tables 1-3 alongside the
    per-entry arrays that Figures 3 and 6 plot.
    """

    n_classes = config['n_classes']
    end_token = n_classes - 1

    samples = samples.long()

    probabilities = marginal_probabilities(samples, n_classes)
    # (n_data, n_classes, seq_len)

    results = evaluate_probabilities(labels.numpy(), probabilities.numpy())

    sampled_times = time_to_event(samples, end_token).float()
    # (n_data, n_samples)

    lower, upper = confidence_interval(sampled_times, alpha)
    results.update(evaluate_intervals(true_times.numpy(), lower.numpy(), upper.numpy(),
                                      sampled_times.numpy()))
    results['alpha'] = alpha

    # FaceMed alone has a known data-generating process, so its estimates can be
    # scored against exact marginals (Equation 9) as well as against samples.
    if config['kind'] == 'facemed' and ages is not None:
        exact = np.stack([ground_truth_marginals(int(a), config['seq_len']) for a in ages])
        # (n_data, n_classes, seq_len)
        per_entry_rmse = np.sqrt(((probabilities.numpy() - exact) ** 2).mean(axis=(0, 1)))
        results['ground_truth_rmse'] = float(per_entry_rmse.mean())
        results['ground_truth_rmse_per_entry'] = per_entry_rmse

    return results


def summarize(results):
    """One-line summary of the sequence-level metrics."""

    fields = ['ECE', 'AUC', 'CE', 'BS', 'coverage', 'relative_width', 'relative_mae']
    parts = [f'{name}[{results[name]:.4f}]' for name in fields if name in results]

    if 'ground_truth_rmse' in results:
        parts.append(f"gt_RMSE[{results['ground_truth_rmse']:.4f}]")

    return ' '.join(parts)


def save_results(results, out_dir, tag):
    """Write sequence-level numbers as JSON and per-entry arrays as .npz."""

    os.makedirs(out_dir, exist_ok=True)

    scalars = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}

    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w') as handle:
        json.dump(scalars, handle, indent=2)

    np.savez_compressed(os.path.join(out_dir, f'per_entry_{tag}.npz'), **arrays)


def main():

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenario', required=True,
                        help='seaquest, riverraid, bank_heist, hero, road_runner or facemed')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--decoder', default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument('--seq_len', default=None, type=int,
                        help='overrides the scenario sequence length; mainly for quick tests')
    parser.add_argument('--n_samples', default=100, type=int,
                        help='Monte Carlo sequences per image (M in Section 3.1)')
    parser.add_argument('--max_parallel_samples', default=None, type=int,
                        help='cap on draws rolled out at once, to bound memory')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--alpha', default=0.9, type=float,
                        help='confidence level for the time-to-event interval')
    parser.add_argument('--condition_first_entry', default=None, type=int,
                        help='evaluate conditional rather than marginal probabilities')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--split_seed', default=42, type=int,
                        help='must match the --split_seed used for training, so the '
                             'test set is the one the model never saw')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config(args.scenario)
    if args.seq_len is not None:
        config = {**config, 'seq_len': args.seq_len}

    data = build_datasets(config, args.data_dir,
                          condition_first_entry=args.condition_first_entry,
                          seed=args.split_seed)
    loader = torch.utils.data.DataLoader(
        data['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = Simulator(config['n_classes'], decoder_type=args.decoder).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    samples, labels, true_times, ages = collect_samples(
        model, loader, config['seq_len'], args.n_samples, device,
        condition_first_entry=args.condition_first_entry,
        max_parallel_samples=args.max_parallel_samples,
    )
    results = evaluate_samples(samples, labels, true_times, config,
                               alpha=args.alpha, ages=ages)

    tag = 'conditional' if args.condition_first_entry is not None else 'marginal'
    print(f'{args.scenario} ({tag}): {summarize(results)}', flush=True)

    save_results(results, args.out_dir, tag)
    torch.save(samples, os.path.join(args.out_dir, f'samples_{tag}.pt'))
    torch.save({'labels': labels, 'true_times': true_times},
               os.path.join(args.out_dir, f'ground_truth_{tag}.pt'))


if __name__ == '__main__':
    main()
