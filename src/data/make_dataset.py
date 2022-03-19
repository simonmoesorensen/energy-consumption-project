import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat

FILE_DIR = Path(__file__).parent
ROOT_DIR = FILE_DIR.parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

TASKIDS = [7805, 8040, 8227, 9289, 10204, 10218]


def load_data(path):
    data = loadmat(path)
    data.pop("__header__")
    data.pop("__version__")
    data.pop("__globals__")
    data = {key: np.squeeze(val) for key, val in data.items()}
    return pd.DataFrame.from_dict(data)


def make_dataset():
    vars = ['acclon', 'traccon', 'vehspd']
    standard_cols = ['alt', 'direction', 'distance', 'lat', 'lon']
    join_col = ['timestamp']
    file_name = 'processed/cph1_data.csv'

    df_tasks = []
    for task in TASKIDS:
        print(f'Creating data for {task=}')
        files = [x for x in DATA_DIR.joinpath('raw').glob('*.mat') if f'_{task}_' in str(x)]

        passes = int(len(files) / len(vars))
        print(f'{passes=}')

        df_passes = []
        for i in range(1, passes + 1):
            print(f'Merging pass={i}')
            pass_files = [x for x in files if f'_pass_{i}' in str(x)]

            # Mapping: ['acclon', 'traccon', 'vehspd']
            df_variables = [load_data(x) for x in pass_files]

            # Get the dataframe with lowest sampling frequency
            min_data = np.argmin([x.shape[0] for x in df_variables])

            # Set root to join on
            root = df_variables[min_data]

            # Get other vars
            others = df_variables[:min_data] + df_variables[min_data + 1:]

            # Join them on root
            for other_df in others:
                root = pd.merge_asof(root, other_df.drop(standard_cols, axis=1), on=join_col)

            root['pass'] = i
            df_passes.append(root)

        # Concatenate all passes of a task
        out_df = pd.concat(df_passes, axis=0)
        out_df['taskID'] = task
        out_df = out_df.reset_index(drop=True)
        df_tasks.append(out_df)

    # Concatenate all tasks
    out_df = pd.concat(df_tasks, axis=0)
    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(DATA_DIR.joinpath(file_name))
    print(f'Saved in data folder with {file_name=}')


if __name__ == '__main__':
    make_dataset()
