from pathlib import Path

import pandas as pd
import numpy as np

FILE_DIR = Path(__file__).parent
ROOT_DIR = FILE_DIR.parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')


def build_features():
    df = pd.read_csv(DATA_DIR.joinpath('processed/cph1_data.csv'), index_col=0)

    # Adjust offsets
    df['traccon'] = df['traccon'] - 160
    df['acclon'] = df['acclon'] - 400

    df.to_csv(DATA_DIR.joinpath('processed/features.csv'))


if __name__ == '__main__':
    build_features()
