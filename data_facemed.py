"""FaceMed: synthetic health trajectories over UTKFace portraits (Appendix B).

Each UTKFace image carries the subject's age in its filename. From that age we
simulate one health-status sequence per subject with an age-dependent Markov
chain, giving image-sequence pairs of exactly the form the framework consumes.
The task is to predict the trajectory from the face alone, so the model has to
infer age from the image and then propagate the correct dynamics.

States are healthy (0), ill (1) and dead (2). Dead is absorbing, so it doubles as
the padding value, and the time-to-event is the index of the first "dead" entry --
the subject's survival time in years.

Transitions follow Figure 5, with the age advancing as the simulation runs:

    age < 40        health never changes
    40 <= age <= 80 healthy <-> ill at 0.1, no death
    age > 80        healthy -> (0.6, 0.4, 0.0), ill -> (0.1, 0.7, 0.2)

Build the dataset once with

    python data_facemed.py --utkface_dir /path/to/UTKFace --out_dir data/facemed

then point train.py at --data_dir data/facemed.
"""

import argparse
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

HEALTHY, ILL, DEAD = 0, 1, 2
N_STATES = 3

# UTKFace filenames look like "25_0_3_20170116174525125.jpg.chip.jpg";
# the leading field is the age.
UTKFACE_NAME = re.compile(r'^(\d+)_\d+_\d+_.*\.(?:jpg|jpeg|png)$', re.IGNORECASE)


def transition_matrix(age):
    """Row-stochastic transition matrix for a subject of the given age.

    Row b, column a is P(Y_i = a | Y_{i-1} = b) of Equation (7).
    """

    if age < 40:
        # Figure 5(a): nothing changes. Every state is absorbing.
        return np.eye(N_STATES)

    if age <= 80:
        # Figure 5(b): healthy and ill interchange, death is not yet possible.
        return np.array([
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ])

    # Figure 5(c): illness becomes more likely and the ill may die.
    return np.array([
        [0.6, 0.4, 0.0],
        [0.1, 0.7, 0.2],
        [0.0, 0.0, 1.0],
    ])


def simulate_sequence(start_age, seq_len, rng):
    """Simulate one health trajectory, starting healthy at `start_age`.

    Entry i is the subject's state in year i, so their age at entry i is
    start_age + i. The transition out of entry i uses the matrix for the age the
    subject has *at* entry i.

    Returns an int array of length seq_len.
    """

    sequence = np.empty(seq_len, dtype=np.int64)
    state = HEALTHY
    sequence[0] = state

    for i in range(1, seq_len):
        probabilities = transition_matrix(start_age + i - 1)[state]
        state = rng.choice(N_STATES, p=probabilities)
        sequence[i] = state

    return sequence


def ground_truth_marginals(start_age, seq_len):
    """Analytic marginals P(Y_i = a | X = x) via Equation (9).

    Because the data-generating process is known, these are exact, and the
    estimates produced by foCus can be scored against them directly -- the RMSE
    check of Section 6.1 and Appendix D.1.

    Returns (n_states, seq_len).
    """

    marginals = np.zeros((N_STATES, seq_len))
    distribution = np.zeros(N_STATES)
    distribution[HEALTHY] = 1.0
    marginals[:, 0] = distribution

    for i in range(1, seq_len):
        distribution = distribution @ transition_matrix(start_age + i - 1)
        marginals[:, i] = distribution

    return marginals


def read_ages(utkface_dir):
    """Collect (filename, age) for every parseable UTKFace image.

    A handful of UTKFace files have malformed names; those are skipped and
    counted rather than crashing the build.
    """

    names, ages, skipped = [], [], 0

    for filename in sorted(os.listdir(utkface_dir)):
        match = UTKFACE_NAME.match(filename)
        if match is None:
            skipped += 1
            continue
        names.append(filename)
        ages.append(int(match.group(1)))

    return names, np.array(ages, dtype=np.int64), skipped


def build(utkface_dir, out_dir, seq_len=100, split_ratio=(0.7, 0.2, 0.1), seed=42):
    """Simulate one sequence per UTKFace image and write the dataset to `out_dir`."""

    rng = np.random.default_rng(seed)

    names, ages, skipped = read_ages(utkface_dir)
    if len(names) == 0:
        raise RuntimeError(f'no UTKFace images found in {utkface_dir}')

    sequences = np.stack([simulate_sequence(age, seq_len, rng) for age in ages])
    # (n_subjects, seq_len)

    # Split over subjects, so a face never appears in two splits.
    order = rng.permutation(len(names))
    n_train = int(round(len(names) * split_ratio[0]))
    n_valid = int(round(len(names) * split_ratio[1]))

    split = np.empty(len(names), dtype=object)
    split[order[:n_train]] = 'train'
    split[order[n_train:n_train + n_valid]] = 'valid'
    split[order[n_train + n_valid:]] = 'test'

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, 'facemed.npz'),
        filenames=np.array(names),
        ages=ages,
        sequences=sequences.astype(np.int8),
        split=split.astype(str),
        image_dir=np.array(utkface_dir),
        seq_len=np.array(seq_len),
    )

    survival = np.where((sequences == DEAD).any(axis=1),
                        (sequences == DEAD).argmax(axis=1), seq_len)

    print(f'subjects            : {len(names)} ({skipped} filenames skipped as malformed)')
    print(f'mean age            : {ages.mean():.2f} years')
    print(f'mean survival time  : {survival.mean():.2f} years')
    print(f'died within {seq_len} years: {(survival < seq_len).mean():.1%}')
    print(f'split               : {(split == "train").sum()} / '
          f'{(split == "valid").sum()} / {(split == "test").sum()}')
    print(f'written             : {os.path.join(out_dir, "facemed.npz")}')


class FaceMedDataset(Dataset):
    """Image-sequence pairs for one split of a built FaceMed dataset."""

    def __init__(self, data_dir, split, image_dir=None, transform=ToTensor()):

        archive = np.load(os.path.join(data_dir, 'facemed.npz'), allow_pickle=False)

        keep = archive['split'] == split
        self.filenames = archive['filenames'][keep]
        self.ages = archive['ages'][keep]
        self.sequences = archive['sequences'][keep].astype(np.int64)
        self.seq_len = int(archive['seq_len'])
        # image_dir is recorded at build time but can be overridden, so a built
        # dataset stays usable after the images move.
        self.image_dir = image_dir or str(archive['image_dir'])
        self.transform = transform

    def __len__(self):

        return len(self.filenames)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.image_dir, self.filenames[index])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        # (3, height, width)

        sequence = self.sequences[index]
        # (seq_len,)

        # Survival time: the first "dead" year, or seq_len if the subject is
        # still alive when the record ends.
        died = np.flatnonzero(sequence == DEAD)
        length = int(died[0]) if len(died) else self.seq_len

        return {
            'image': image,
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'length': length,
            'age': int(self.ages[index]),
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--utkface_dir', required=True,
                        help='directory of UTKFace images (the aligned and cropped release)')
    parser.add_argument('--out_dir', default='data/facemed')
    parser.add_argument('--seq_len', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    build(args.utkface_dir, args.out_dir, seq_len=args.seq_len, seed=args.seed)
