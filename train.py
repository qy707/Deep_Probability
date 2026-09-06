"""Train the autoregressive simulator (Section 3.2, Appendix C.2).

One run trains one simulator on one scenario with one regularization setting. The
three variants compared throughout the paper differ only in the lambda schedule
of Equation (4), so they are selected with a single flag:

    --regularization none             the maximum-likelihood baseline of Section 5
    --regularization time_dependent   the proposed schedule of Section 6 (Table 5)
    --regularization constant         the same lambda at every entry (Table 5)

Example:

    python train.py --scenario seaquest --data_dir /data/Atari_HEAD \
        --regularization time_dependent --out_dir runs/seaquest_time_dependent

Every `--eval_interval` epochs the simulator is sampled and scored with the full
Monte Carlo framework, so the run leaves behind the learning curves of Figures 13
to 15 alongside its checkpoints.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from configs import get_config, lambda_schedule
from datasets import build_datasets
from evaluate import collect_samples, evaluate_samples, save_results, summarize
from losses import SequenceLoss
from simulator import Simulator


def teacher_forced_logits(model, batch, device):
    """Run the simulator over a batch with the ground-truth past (Figure 2a).

    Returns (logits, targets) shaped for a cross-entropy loss:
    (n_batch, n_classes, seq_len) and (n_batch, seq_len).
    """

    images = batch['image'].to(device)
    # (n_batch, 3, height, width)
    targets = batch['sequence'].to(device)
    # (n_batch, seq_len)

    prefix = targets[:, :-1].t().contiguous()
    # (seq_len - 1, n_batch); the image supplies the input for the first entry

    logits = model(images, prefix)
    # (seq_len, n_batch, n_classes)

    return logits.permute(1, 2, 0), targets
    # (n_batch, n_classes, seq_len), (n_batch, seq_len)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains when an optimizer is given, else evaluates.

    Returns (mean loss per sequence, mean cross entropy per entry).
    """

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_entry_ce = None
    n_sequences = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            logits, targets = teacher_forced_logits(model, batch, device)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_batch = targets.size(0)
            total_loss += loss.item() * n_batch
            n_sequences += n_batch

            with torch.no_grad():
                entry_ce = F.cross_entropy(logits, targets, reduction='none').sum(dim=0)
                # (seq_len,)
                total_entry_ce = entry_ce if total_entry_ce is None else total_entry_ce + entry_ce

    return total_loss / n_sequences, (total_entry_ce / n_sequences).cpu().numpy()


def main():

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenario', required=True,
                        help='seaquest, riverraid, bank_heist, hero, road_runner or facemed')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--regularization', default='time_dependent',
                        choices=['none', 'time_dependent', 'constant'])
    parser.add_argument('--lambdas', default=None,
                        help='comma-separated lambda_1,lambda_2,... overriding '
                             '--regularization; later entries get 0. Use this for the '
                             'single-entry schedules of the Appendix E sensitivity study, '
                             'e.g. --lambdas 0,0,0,0.1')
    parser.add_argument('--decoder', default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--seq_len', default=None, type=int,
                        help='overrides the scenario sequence length; mainly for quick tests')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=None, type=float,
                        help='overrides the scenario default from Appendix C.2')
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--n_samples', default=100, type=int,
                        help='Monte Carlo sequences per image (M in Section 3.1)')
    parser.add_argument('--max_parallel_samples', default=None, type=int,
                        help='cap on draws rolled out at once, to bound memory')
    parser.add_argument('--eval_interval', default=5, type=int)
    parser.add_argument('--eval_split', default='valid', choices=['valid', 'test'],
                        help="split sampled during training; 'valid' keeps the test "
                             "set untouched for the final evaluate.py run")
    parser.add_argument('--alpha', default=0.9, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int,
                        help='model seed; vary it for independent realizations')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='seed for the train/valid/test split. Keep this fixed '
                             'across model seeds, or different realizations would be '
                             'trained and scored on different data')
    args = parser.parse_args()

    config = get_config(args.scenario)
    if args.seq_len is not None:
        config = {**config, 'seq_len': args.seq_len}
    learning_rate = args.lr if args.lr is not None else config['learning_rate']
    if args.lambdas is not None:
        lambdas = lambda_schedule(config, [float(v) for v in args.lambdas.split(',')])
    else:
        lambdas = lambda_schedule(config, args.regularization)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as handle:
        json.dump({**vars(args), 'learning_rate': learning_rate, 'lambdas': lambdas},
                  handle, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f'scenario       : {args.scenario} ({config["kind"]}, {config["n_classes"]} classes, '
          f'sequence length {config["seq_len"]})', flush=True)
    print(f'regularization : {args.regularization}; '
          f'nonzero lambda at {int(np.count_nonzero(lambdas))} of {len(lambdas)} entries',
          flush=True)
    print(f'decoder        : {args.decoder}, lr {learning_rate}, device {device}', flush=True)

    data = build_datasets(config, args.data_dir, seed=args.split_seed)
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == 'train'),
            num_workers=args.num_workers,
        )
        for split, dataset in data.items()
    }
    for split, dataset in data.items():
        print(f'{split:<6} size    : {len(dataset)}', flush=True)

    model = Simulator(
        config['n_classes'], decoder_type=args.decoder, embed_size=args.embed_size,
        hidden_size=args.hidden_size, n_heads=args.n_heads, num_layers=args.num_layers,
    ).to(device)
    print(f'parameters     : {sum(p.numel() for p in model.parameters() if p.requires_grad)}',
          flush=True)

    criterion = SequenceLoss(lambdas).to(device)
    # Appendix C.2: Adam, no weight decay, constant learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(1, args.epochs + 1):

        train_loss, train_entry_ce = run_epoch(model, loaders['train'], criterion,
                                               device, optimizer)
        valid_loss, valid_entry_ce = run_epoch(model, loaders['valid'], criterion, device)

        record = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss}
        print(f'epoch {epoch:3d}  train_loss {train_loss:8.3f}  valid_loss {valid_loss:8.3f}',
              flush=True)

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            samples, labels, true_times, ages = collect_samples(
                model, loaders[args.eval_split], config['seq_len'], args.n_samples, device,
                max_parallel_samples=args.max_parallel_samples,
            )
            results = evaluate_samples(samples, labels, true_times, config,
                                       alpha=args.alpha, ages=ages)

            print(f'          {args.eval_split}: {summarize(results)}', flush=True)

            save_results(results, os.path.join(args.out_dir, 'eval'), f'epoch_{epoch}')
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pt'))

            record.update({k: v for k, v in results.items() if not isinstance(v, np.ndarray)})

        history.append(record)
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as handle:
            json.dump(history, handle, indent=2)

        np.save(os.path.join(args.out_dir, 'train_entry_cross_entropy.npy'), train_entry_ce)
        np.save(os.path.join(args.out_dir, 'valid_entry_cross_entropy.npy'), valid_entry_ce)

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'checkpoint_final.pt'))
    print(f'done; wrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
