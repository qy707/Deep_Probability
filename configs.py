"""Dataset and regularization settings for every scenario in the paper.

Each scenario is one dict, so that everything needed to reproduce a run lives in
one readable place. The fields map onto the paper as follows:

  n_classes, seq_len, frame_stride        Table 4 and Appendix B
  learning_rate                           Appendix C.2
  time_dependent_lambdas                  Table 5, "Time-dependent lambda's"
  constant_lambda                         Table 5, "Constant lambda's"
  condition_first_entry                   Appendix D.2 (event conditioned on)

The regularization coefficients are the lambda_i of Equation (4): the strength of
the logit penalty applied when predicting entry i. `time_dependent_lambdas` lists
lambda_1, lambda_2, ... for the leading entries; every later entry gets 0.
"""


# --- Atari-HEAD video games (Section 4.2, Table 4) -------------------------------
#
# n_classes = 19: the 18 Atari actions plus an "end of game sequence" token.
# The end token is always the last class, so end_token = n_classes - 1 = 18.
# frame_stride is the number of game frames per sequence entry, so an entry index
# i corresponds to t = frame_stride / 60 Hz * i seconds of gameplay.

seaquest_config = {
    'name': 'seaquest',
    'kind': 'atari',
    'n_classes': 19,
    'seq_len': 200,
    'frame_stride': 3,
    'learning_rate': 1e-5,
    'split_ratio': [0.75, 0.125, 0.125],
    'condition_first_entry': 0,                          # NOOP
    'time_dependent_lambdas': [0.05, 0.01, 0.05],        # Table 5: lambda_1:3
    'constant_lambda': 0.001,
}

riverraid_config = {
    'name': 'riverraid',
    'kind': 'atari',
    'n_classes': 19,
    'seq_len': 300,
    'frame_stride': 2,
    'learning_rate': 1e-5,
    'split_ratio': [0.75, 0.125, 0.125],
    'condition_first_entry': 0,                          # NOOP
    'time_dependent_lambdas': [0.01] * 6,                # Table 5: lambda_1:6
    'constant_lambda': 0.001,
}

bank_heist_config = {
    'name': 'bank_heist',
    'kind': 'atari',
    'n_classes': 19,
    'seq_len': 300,
    'frame_stride': 2,
    'learning_rate': 1e-5,
    'split_ratio': [0.75, 0.125, 0.125],
    'condition_first_entry': 3,                          # RIGHT
    'time_dependent_lambdas': [0.05] + [0.01] * 10,      # Table 5: lambda_1, lambda_2:11
    'constant_lambda': 0.001,
}

hero_config = {
    'name': 'hero',
    'kind': 'atari',
    'n_classes': 19,
    'seq_len': 300,
    'frame_stride': 2,
    'learning_rate': 5e-5,
    'split_ratio': [0.75, 0.125, 0.125],
    'condition_first_entry': 0,                          # NOOP
    'time_dependent_lambdas': [0.01] + [0.005] * 5,      # Table 5: lambda_1, lambda_2:6
    'constant_lambda': 0.001,
}

road_runner_config = {
    'name': 'road_runner',
    'kind': 'atari',
    'n_classes': 19,
    'seq_len': 300,
    'frame_stride': 4,
    'learning_rate': 1e-5,
    'split_ratio': [0.7, 0.2, 0.1],
    'condition_first_entry': 4,                          # LEFT
    'time_dependent_lambdas': [0.01] + [0.005] * 20,     # Table 5: lambda_1, lambda_2:21
    'constant_lambda': 0.001,
}


# --- FaceMed synthetic health trajectories (Appendix B) --------------------------
#
# n_classes = 3: healthy (0), ill (1), dead (2). Dead is both the absorbing state
# and the padding value, so end_token = n_classes - 1 = 2 and the time-to-event is
# the index of the first "dead" entry, i.e. the subject's survival time in years.

facemed_config = {
    'name': 'facemed',
    'kind': 'facemed',
    'n_classes': 3,
    'seq_len': 100,                                      # one entry per year
    'learning_rate': 1e-5,
    'split_ratio': [0.7, 0.2, 0.1],
    'condition_first_entry': 0,                          # Healthy in year 1
    # Table 5 prints "lambda_4:5 = 0.005  lambda_5:50 = 0.001"; index 5 appears
    # twice, which we read as lambda_6:50 = 0.001.
    'time_dependent_lambdas': [0.01] * 3 + [0.005] * 2 + [0.001] * 45,
    'constant_lambda': 0.001,
}


ALL_CONFIGS = {
    cfg['name']: cfg
    for cfg in [
        seaquest_config,
        riverraid_config,
        bank_heist_config,
        hero_config,
        road_runner_config,
        facemed_config,
    ]
}


def get_config(name):
    """Look up a scenario config by name, with a helpful error if it is unknown."""

    if name not in ALL_CONFIGS:
        raise KeyError(f'unknown scenario {name!r}; choose one of {sorted(ALL_CONFIGS)}')

    return ALL_CONFIGS[name]


def lambda_schedule(config, regularization):
    """Build the full per-entry lambda vector of Equation (4).

    regularization is one of:
      'none'            no regularization, the maximum-likelihood baseline of Section 5
      'time_dependent'  the proposed schedule of Section 6 (Table 5, left column)
      'constant'        the same lambda at every entry (Table 5, right column)

    Returns a list of seq_len floats.
    """

    seq_len = config['seq_len']

    # An explicit list is taken as the schedule itself, padded with zeros. This is
    # how the single-entry schedules of the Appendix E sensitivity study are set.
    if isinstance(regularization, (list, tuple)):
        given = list(regularization)[:seq_len]
        return given + [0.0] * (seq_len - len(given))

    if regularization == 'none':
        return [0.0] * seq_len

    if regularization == 'constant':
        return [config['constant_lambda']] * seq_len

    if regularization == 'time_dependent':
        leading = list(config['time_dependent_lambdas'])[:seq_len]
        return leading + [0.0] * (seq_len - len(leading))

    raise ValueError(
        f'unknown regularization {regularization!r}; '
        "choose 'none', 'time_dependent' or 'constant'"
    )
