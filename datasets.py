"""Build train/valid/test datasets for a scenario.

The two data sources look different on disk but produce the same sample dict, so
everything downstream is source-agnostic:

    {'image': (3, H, W) float tensor,
     'sequence': (seq_len,) int64 tensor of class indices,
     'length': int, the time-to-event of the ground-truth sequence}

`condition_first_entry` restricts the test set to sequences whose first entry
takes a given value, which is how the conditional probabilities of Table 2 are
evaluated (Appendix D.2). It is applied to the test split only; training always
sees every sequence.
"""

import os

from data_atari import AtariTrialDataset, split_trials
from data_facemed import FaceMedDataset
from torch.utils.data import ConcatDataset


def build_datasets(config, data_dir, condition_first_entry=None, seed=42):
    """Return {'train': Dataset, 'valid': Dataset, 'test': Dataset} for a scenario."""

    if config['kind'] == 'facemed':
        datasets = {
            split: FaceMedDataset(data_dir, split)
            for split in ('train', 'valid', 'test')
        }

        # The sequence length is baked into the built dataset. A mismatch with the
        # config would otherwise surface much later as a confusing shape error.
        built = datasets['train'].seq_len
        if built != config['seq_len']:
            raise ValueError(
                f'{data_dir} was built with seq_len={built} but the config expects '
                f"{config['seq_len']}; rebuild the dataset or pass --seq_len {built}"
            )

        return datasets

    if config['kind'] == 'atari':
        game_dir = os.path.join(data_dir, config['name'])
        trials = split_trials(data_dir, config['name'], config['split_ratio'], seed=seed)

        datasets = {}
        for split, trial_names in trials.items():
            # Only the test split is conditioned; train and valid stay complete.
            condition = condition_first_entry if split == 'test' else None
            datasets[split] = ConcatDataset([
                AtariTrialDataset(
                    game_dir, trial_name,
                    seq_len=config['seq_len'],
                    frame_stride=config['frame_stride'],
                    end_token=config['n_classes'] - 1,
                    condition_first_entry=condition,
                )
                for trial_name in trial_names
            ])

        return datasets

    raise ValueError(f"unknown dataset kind {config['kind']!r}")
