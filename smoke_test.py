"""End-to-end check that runs without downloading any data.

Builds a miniature FaceMed dataset from randomly generated "portraits", trains
the simulator on it for a few epochs, draws Monte Carlo sequences, and scores
them with every metric in the paper. It exercises the whole path -- config,
dataset, simulator, loss, Monte Carlo estimation, metrics -- for both the RNN and
the Transformer decoder, and checks that the logit regularizer does what
Section 6 says it does.

    python smoke_test.py

The numbers it prints are meaningless as science; the point is that each stage
runs, the shapes line up, and the metrics land in their valid ranges.
"""

import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from configs import facemed_config, lambda_schedule
from data_atari import AtariTrialDataset, TrajectoryReader, as_boolean, split_trials
from data_facemed import DEAD, build, ground_truth_marginals, simulate_sequence
from datasets import build_datasets
from evaluate import collect_samples, evaluate_samples, summarize
from losses import SequenceLoss
from monte_carlo import (confidence_interval, conditional_probabilities,
                         marginal_probabilities, time_to_event)
from simulator import Simulator
from train import run_epoch

N_SUBJECTS = 96
SEQ_LEN = 12
N_SAMPLES = 24
IMAGE_SIZE = 32

passed, failed = [], []


def check(name, condition, detail=''):
    """Record one assertion without aborting the rest of the run."""

    (passed if condition else failed).append(name)
    print(f'  [{"PASS" if condition else "FAIL"}] {name}{"  " + detail if detail else ""}')


def make_fake_utkface(directory, rng):
    """Write random images under UTKFace's `age_gender_race_stamp.jpg` naming."""

    os.makedirs(directory, exist_ok=True)
    for i in range(N_SUBJECTS):
        age = int(rng.integers(1, 100))
        pixels = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(os.path.join(directory, f'{age}_0_0_2017{i:08d}.jpg'))


def make_fake_atari(root, rng, n_trials=3, n_frames=12):
    """Write a miniature Atari-HEAD tree: trajectory files, packed frames, metadata.

    Mirrors the real layout closely enough to exercise the loader -- one scoring
    event every four frames, frames packed as <trial>/<frame_id>.png inside a
    .tar.bz2, and a meta_data.csv naming the usable trials.
    """

    game_dir = os.path.join(root, 'seaquest')
    os.makedirs(game_dir, exist_ok=True)
    meta_rows = []

    for trial in range(1, n_trials + 1):
        trial_name = f'{trial}_RZ_{1000 + trial}_Jul-15'
        lines = ['frame_id,episode_id,score,duration(ms),unclipped_reward,action']
        staging = os.path.join(game_dir, trial_name)
        os.makedirs(staging, exist_ok=True)

        for frame in range(n_frames):
            frame_id = f'{trial_name}_0_{frame}'
            reward = 1 if (frame + 1) % 4 == 0 else 0
            lines.append(f'{frame_id},0,0,50,{reward},{int(rng.integers(0, 18))}')
            pixels = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(os.path.join(staging, frame_id + '.png'))

        with open(os.path.join(game_dir, trial_name + '.txt'), 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
        with tarfile.open(os.path.join(game_dir, trial_name + '.tar.bz2'), 'w:bz2') as tar:
            tar.add(staging, arcname=trial_name)
        shutil.rmtree(staging)          # the loader must unpack it itself

        meta_rows.append(f'seaquest,R,False,{trial}')

    with open(os.path.join(root, 'meta_data.csv'), 'w') as handle:
        handle.write('GameName,subject_id,frame_averaging,trial_id\n'
                     + '\n'.join(meta_rows) + '\n')


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    workspace = tempfile.mkdtemp(prefix='focus_smoke_')

    try:
        print('\n1. FaceMed simulation (Appendix B)')
        exact = ground_truth_marginals(85, SEQ_LEN)
        drawn = np.stack([simulate_sequence(85, SEQ_LEN, rng) for _ in range(4000)])
        empirical = np.stack([(drawn == s).mean(0) for s in range(3)])
        gap = np.abs(empirical - exact).max()
        check('simulated marginals match the analytic Equation (9) marginals',
              gap < 0.05, f'max gap {gap:.4f}')
        check('everyone starts healthy', bool((drawn[:, 0] == 0).all()))
        check('death is absorbing',
              all(bool((row[np.argmax(row == DEAD):] == DEAD).all())
                  for row in drawn if (row == DEAD).any()))

        print('\n2. Dataset build and loading')
        image_dir = os.path.join(workspace, 'utkface')
        make_fake_utkface(image_dir, rng)
        build(image_dir, os.path.join(workspace, 'facemed'), seq_len=SEQ_LEN, seed=0)

        config = {**facemed_config, 'seq_len': SEQ_LEN}
        data = build_datasets(config, os.path.join(workspace, 'facemed'))
        check('all three splits are non-empty',
              all(len(d) > 0 for d in data.values()),
              ' / '.join(f'{k}={len(v)}' for k, v in data.items()))

        sample = data['train'][0]
        check('sample has the expected shapes',
              sample['image'].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
              and sample['sequence'].shape == (SEQ_LEN,),
              f"image {tuple(sample['image'].shape)}, sequence {tuple(sample['sequence'].shape)}")

        print('\n3. Monte Carlo estimation (Section 3.1)')
        fake = torch.randint(0, 3, (7, N_SAMPLES, SEQ_LEN))
        probabilities = marginal_probabilities(fake, 3)
        check('marginal probabilities sum to one over classes',
              torch.allclose(probabilities.sum(1), torch.ones(7, SEQ_LEN)))
        conditional, n_matching = conditional_probabilities(fake, 3, 0, 1)
        check('conditional probabilities sum to one where the condition is met',
              torch.allclose(conditional[n_matching > 0].sum(1),
                             torch.ones(int((n_matching > 0).sum()), SEQ_LEN)))
        times = time_to_event(fake, DEAD)
        check('time-to-event lies within the sequence', bool((times >= 0).all() and (times <= SEQ_LEN).all()))
        lower, upper = confidence_interval(times.float(), 0.9)
        check('confidence interval is ordered', bool((lower <= upper).all()))

        print('\n4. Training and evaluation, RNN decoder (Section 3.2)')
        loaders = {
            split: torch.utils.data.DataLoader(dataset, batch_size=16,
                                               shuffle=(split == 'train'), num_workers=0)
            for split, dataset in data.items()
        }
        model = Simulator(3, decoder_type='rnn', embed_size=32, hidden_size=32)
        criterion = SequenceLoss(lambda_schedule(config, 'time_dependent'))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        first_loss, _ = run_epoch(model, loaders['train'], criterion, 'cpu', optimizer)
        for _ in range(3):
            last_loss, _ = run_epoch(model, loaders['train'], criterion, 'cpu', optimizer)
        check('training loss decreases', last_loss < first_loss,
              f'{first_loss:.3f} -> {last_loss:.3f}')

        samples, labels, true_times, ages = collect_samples(
            model, loaders['test'], SEQ_LEN, N_SAMPLES, 'cpu')
        check('sampled sequences have the expected shape',
              samples.shape == (len(data['test']), N_SAMPLES, SEQ_LEN),
              str(tuple(samples.shape)))

        results = evaluate_samples(samples, labels, true_times, config, ages=ages)
        print(f'       {summarize(results)}')
        check('ECE is a probability', 0.0 <= results['ECE'] <= 1.0)
        check('AUC is in [0, 1]', 0.0 <= results['AUC'] <= 1.0)
        check('Brier score is in [0, 2]', 0.0 <= results['BS'] <= 2.0)
        check('cross entropy is non-negative', results['CE'] >= 0.0)
        check('coverage is a probability', 0.0 <= results['coverage'] <= 1.0)
        check('every sequence-level metric is finite',
              all(np.isfinite(v) for v in results.values() if isinstance(v, float)))
        check('FaceMed ground-truth RMSE is computed (Section 6.1)',
              'ground_truth_rmse' in results and 0.0 <= results['ground_truth_rmse'] <= 1.0,
              f"{results.get('ground_truth_rmse', float('nan')):.4f}")

        print('\n5. Conditioning the first entry (Table 2, Appendix D.2)')
        conditioned = model.sample_sequences(
            torch.stack([data['test'][i]['image'] for i in range(4)]),
            SEQ_LEN, N_SAMPLES, condition_first_entry=1)
        check('every conditioned sample starts at the requested state',
              bool((conditioned[:, :, 0] == 1).all()))

        print('\n6. Transformer decoder (Appendix G)')
        transformer = Simulator(3, decoder_type='transformer', embed_size=32, n_heads=4)
        t_first, _ = run_epoch(transformer, loaders['train'], criterion, 'cpu',
                               torch.optim.Adam(transformer.parameters(), lr=1e-3))
        t_samples = transformer.sample_sequences(sample['image'].unsqueeze(0), SEQ_LEN, N_SAMPLES)
        check('transformer trains and samples',
              np.isfinite(t_first) and t_samples.shape == (1, N_SAMPLES, SEQ_LEN),
              f'loss {t_first:.3f}, samples {tuple(t_samples.shape)}')

        images = torch.stack([data['train'][i]['image'] for i in range(4)])
        entries = torch.stack([data['train'][i]['sequence'] for i in range(4)])
        prefix = entries[:, :-1].t().contiguous()

        # The sampler decodes one entry at a time through a cache, while training
        # runs the whole sequence at once behind a causal mask. If those two paths
        # ever disagreed, the simulator would be sampling from a different
        # distribution than the one it was trained to fit -- so check they match
        # numerically, for both decoders.
        for name, decoder in [('rnn', model), ('transformer', transformer)]:
            decoder.eval()
            with torch.no_grad():
                full = decoder(images, prefix)
                features = decoder.encoder(images)
                decoder_input, state = decoder.decoder.start(features)
                stepwise = []
                for i in range(SEQ_LEN):
                    logits, state = decoder.decoder.step(decoder_input, state, i)
                    stepwise.append(logits)
                    decoder_input = decoder.decoder.embed_entry(entries[:, i])
                stepwise = torch.stack(stepwise)
            gap = (full - stepwise).abs().max().item()
            check(f'{name}: incremental rollout reproduces the teacher-forced pass',
                  gap < 1e-4, f'max gap {gap:.1e}')

        print('\n7. The regularizer suppresses logit magnitude (Section 6)')
        norms = {}
        for name, regularization in [('unregularized', 'none'), ('regularized', 'constant')]:
            torch.manual_seed(1)
            probe = Simulator(3, decoder_type='rnn', embed_size=32, hidden_size=32)
            probe_loss = SequenceLoss([0.0] * SEQ_LEN if regularization == 'none'
                                      else [0.5] * SEQ_LEN)
            probe_optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
            for _ in range(4):
                run_epoch(probe, loaders['train'], probe_loss, 'cpu', probe_optimizer)
            with torch.no_grad():
                logits = probe(images, prefix)
            norms[name] = torch.linalg.vector_norm(logits, dim=-1).mean().item()
        check('a positive lambda shrinks the logit norm',
              norms['regularized'] < norms['unregularized'],
              f"{norms['unregularized']:.3f} -> {norms['regularized']:.3f}")

        print('\n8. Atari-HEAD loader (Section 4.2)')
        atari_root = os.path.join(workspace, 'atari')
        make_fake_atari(atari_root, rng)

        trials = split_trials(atari_root, 'seaquest', [0.75, 0.125, 0.125])
        assigned = [name for names in trials.values() for name in names]
        check('trial split is disjoint and covers every trial',
              len(assigned) == len(set(assigned)) == 3,
              ' / '.join(f'{k}={len(v)}' for k, v in trials.items()))
        check('trial split is deterministic',
              split_trials(atari_root, 'seaquest', [0.75, 0.125, 0.125]) == trials)
        check('frame_averaging is read as a boolean whatever its spelling',
              (~as_boolean(pd.Series(['no', 'yes']))).tolist() == [True, False])

        trial_name = trials['train'][0]
        records = TrajectoryReader(
            os.path.join(atari_root, 'seaquest', trial_name + '.txt')).scoring_sequences()
        check('every frame lands in a scoring sequence', len(records) == 12, f'{len(records)} frames')
        check('the offset into the action series restarts at each scoring event',
              int(records['series_index'].max()) == 3)

        atari = AtariTrialDataset(os.path.join(atari_root, 'seaquest'), trial_name,
                                  seq_len=8, frame_stride=1, end_token=18)
        item = atari[0]
        check('frames are unpacked from the archive on first use',
              item['image'].shape == (3, IMAGE_SIZE, IMAGE_SIZE), str(tuple(item['image'].shape)))
        check('sequences are padded with the end token past their true length',
              bool((item['sequence'][item['length']:] == 18).all()) and 1 <= item['length'] <= 8,
              f"length {item['length']}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print(f'\n{len(passed)} passed, {len(failed)} failed')
    if failed:
        for name in failed:
            print(f'  FAILED: {name}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
