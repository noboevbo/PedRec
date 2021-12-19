import math
import numpy as np
import pandas as pd

from pedrec.configs.dataset_configs import PedRecDatasetConfig
from pedrec.models.constants.sample_method import SAMPLE_METHOD


def get_subsampled_df(df_path: str, cfg: PedRecDatasetConfig):
    df = pd.read_pickle(df_path)

    df_full_length = len(df)
    if cfg.subsample != 1:
        if cfg.subsampling_strategy == SAMPLE_METHOD.SYSTEMATIC:
            df = df.loc[range(0, df_full_length, cfg.subsample)]
        elif cfg.subsampling_strategy == SAMPLE_METHOD.RANDOM:
            df = df.loc[np.random.choice(df.index, math.floor(len(df) / cfg.subsample), replace=False)]
        else:
            raise NotImplementedError(f"Sampling strategy {cfg.subsampling_strategy.name} is not implemented.")
    return df