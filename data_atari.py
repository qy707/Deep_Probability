"""Atari-HEAD gameplay sequences (Section 4.2, Appendix B).

Atari-HEAD records one human keyboard action per game frame. We turn each frame
into an input image and the actions that follow it, up to the next scoring event,
into the sequence to predict. Because consecutive frames usually repeat an action,
entries are subsampled every `frame_stride` frames, so entry i sits at
t = frame_stride / 60 Hz * i seconds of gameplay.

Sequences are padded to `seq_len` with an "end of game sequence" token, which is
the last class index (18 of 19). The time-to-event is the index of the first end
token, i.e. the time until the player next scores.

Expected layout, matching the Atari-HEAD release:

    <data_dir>/meta_data.csv
    <data_dir>/<game>/<trial>.txt          one trajectory per trial
    <data_dir>/<game>/<trial>.tar.bz2      the matching frames
    <data_dir>/<game>/<trial>/             extracted frames (created on first use)

Trials are split into train/valid/test, so frames from one gameplay session never
straddle two splits.
"""

import os
import tarfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

# The first six columns of an Atari-HEAD trajectory file.
TRAJECTORY_COLUMNS = 6


class TrajectoryReader:
    """Parse one Atari-HEAD trajectory file into per-frame action sequences."""

    def __init__(self, trajectory_path, episode_csv=None):

        self.trajectory_path = trajectory_path
        self.episode_csv = episode_csv
        self.trial_id = int(os.path.basename(trajectory_path).split('_')[0])

        with open(trajectory_path, 'r') as handle:
            self.records = handle.readlines()

    def to_dataframe(self):
        """Read the trajectory into a frame-indexed table, dropping frames with
        no recorded action."""

        column_names = self.records[0].rstrip('\n').split(',')[:TRAJECTORY_COLUMNS]
        rows = []

        for line in self.records[1:]:
            fields = line.rstrip('\n').split(',')[:TRAJECTORY_COLUMNS]
            # The frame id is the image filename; the rest are numeric.
            row = [fields[0] + '.png']
            row += [np.nan if value == 'null' else float(value) for value in fields[1:]]
            rows.append(row)

        frame = pd.DataFrame(rows, columns=column_names)

        if 'episode_id' not in frame.columns or frame['episode_id'].isna().all():
            frame['episode_id'] = self._episode_ids_from_csv(len(frame))

        frame = frame[pd.notnull(frame['action'])]

        return frame.reset_index(drop=True)

    def _episode_ids_from_csv(self, n_frames):
        """Recover episode boundaries for the few trials whose trajectory file has
        no usable episode_id column, from an optional <game>_episodes.csv listing
        the starting frame of each episode."""

        if self.episode_csv is None or not os.path.exists(self.episode_csv):
            raise RuntimeError(
                f'{self.trajectory_path} has no usable episode_id column and no '
                f'episode table was found at {self.episode_csv}. Either supply one '
                'or exclude this trial.'
            )

        table = pd.read_csv(self.episode_csv)
        row = table[table['trial_id'] == self.trial_id]
        total = row['total_episode'].values[0]
        starts = row[[f'episode_start_{i + 1}' for i in range(total)]].values[0]

        episode_ids = []
        for i, start in enumerate(starts):
            end = n_frames if i == len(starts) - 1 else starts[i + 1] - 1
            episode_ids.append(np.full(end - start + 1, i))

        return np.concatenate(episode_ids)

    def scoring_sequences(self, condition_first_entry=None):
        """Split each episode at its scoring events.

        Every frame becomes one training example whose target is the actions from
        that frame through the next scoring event. If `condition_first_entry` is
        given, only frames whose own action equals it are kept -- this is how the
        conditional-probability test set of Appendix D.2 is formed.
        """

        trajectory = self.to_dataframe()
        pieces = []

        for episode_id in trajectory['episode_id'].dropna().unique():
            in_episode = trajectory['episode_id'] == episode_id
            episode_start = trajectory[in_episode].index[0]
            scoring_frames = trajectory[in_episode & (trajectory['unclipped_reward'] != 0)].index

            for i, end in enumerate(scoring_frames):
                start = episode_start if i == 0 else scoring_frames[i - 1] + 1
                piece = trajectory.loc[start:end].copy()

                actions = piece['action'].values
                # Every frame in the piece shares the same action series; its
                # own offset into that series says where its target begins.
                piece['action_series'] = [actions] * len(piece)
                piece['series_index'] = np.arange(len(piece))
                pieces.append(piece)

        if not pieces:
            return pd.DataFrame(columns=list(trajectory.columns) + ['action_series', 'series_index'])

        records = pd.concat(pieces, ignore_index=True)

        if condition_first_entry is not None:
            records = records[records['action'] == condition_first_entry].reset_index(drop=True)

        return records


class AtariTrialDataset(Dataset):
    """Image-sequence pairs from a single Atari-HEAD trial."""

    def __init__(self, game_dir, trial_name, seq_len, frame_stride, end_token,
                 condition_first_entry=None, transform=ToTensor()):

        self.game_dir = game_dir
        self.trial_name = trial_name
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.end_token = end_token
        self.transform = transform

        game_name = os.path.basename(os.path.normpath(game_dir))
        reader = TrajectoryReader(
            os.path.join(game_dir, trial_name + '.txt'),
            episode_csv=os.path.join(game_dir, f'{game_name}_episodes.csv'),
        )
        self.records = reader.scoring_sequences(condition_first_entry)

        self._extract_frames_once()

    def _extract_frames_once(self):
        """Unpack this trial's frame archive the first time it is needed."""

        frame_dir = os.path.join(self.game_dir, self.trial_name)
        if os.path.isdir(frame_dir):
            return

        archive = os.path.join(self.game_dir, self.trial_name + '.tar.bz2')
        with tarfile.open(archive, 'r:bz2') as tar:
            tar.extractall(self.game_dir)

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records.iloc[index]

        image = Image.open(os.path.join(self.game_dir, self.trial_name, record['frame_id']))
        image = rgb_to_grayscale(image, num_output_channels=3)

        if self.transform is not None:
            image = self.transform(image)
        # (3, 210, 160)

        # Actions from this frame to the next scoring event, every frame_stride frames.
        future = record['action_series'][record['series_index']::self.frame_stride]
        length = min(len(future), self.seq_len)

        sequence = np.full(self.seq_len, self.end_token, dtype=np.int64)
        sequence[:length] = future[:length]

        return {
            'image': image,
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'length': length,
        }


def as_boolean(column):
    """Interpret a metadata column as booleans, whatever spelling the CSV uses.

    pandas types this column as bool for True/False and as int for 0/1, but as
    plain strings for spellings like yes/no -- and `astype(bool)` on a non-empty
    string is always True, which would silently select the wrong trials. Coerce
    explicitly and refuse anything unrecognized.
    """

    if column.dtype == bool:
        return column

    if pd.api.types.is_numeric_dtype(column):
        return column.astype(bool)

    text = column.astype(str).str.strip().str.lower()
    truthy, falsy = {'true', '1', 'yes', 't'}, {'false', '0', 'no', 'f'}

    unknown = set(text.unique()) - truthy - falsy
    if unknown:
        raise ValueError(f'cannot read frame_averaging values as booleans: {sorted(unknown)}')

    return text.isin(truthy)


def split_trials(data_dir, game_name, split_ratio, seed=42):
    """Assign this game's trials to train/valid/test.

    Only trials listed in meta_data.csv for subject 'R' without frame averaging
    are used, matching the Atari-HEAD subset the paper trains on. The split is
    driven by an explicit seeded generator so it is identical across runs and
    across scripts.
    """

    game_dir = os.path.join(data_dir, game_name)
    meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))

    wanted = meta[
        (meta['GameName'] == game_name)
        & (meta['subject_id'] == 'R')
        & (~as_boolean(meta['frame_averaging']))
    ]['trial_id'].values

    available = {}
    for filename in os.listdir(game_dir):
        if filename.endswith('.txt'):
            available[int(filename.split('_')[0])] = filename[:-4]

    trial_ids = sorted(set(wanted).intersection(available))
    if not trial_ids:
        raise RuntimeError(f'no usable trials for {game_name} in {game_dir}')

    order = np.random.default_rng(seed).permutation(len(trial_ids))
    n_train = int(round(len(trial_ids) * split_ratio[0]))
    n_valid = int(round(len(trial_ids) * split_ratio[1]))

    chosen = {
        'train': order[:n_train],
        'valid': order[n_train:n_train + n_valid],
        'test': order[n_train + n_valid:],
    }

    return {
        split: [available[trial_ids[i]] for i in indices]
        for split, indices in chosen.items()
    }
