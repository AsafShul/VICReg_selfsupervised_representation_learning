import os
import torch
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', type=str, default='./')
args = parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print('Using device: ', device)
    return device


def get_wandb_key():
    return os.environ['WANDB_API_KEY'] if 'WANDB_API_KEY' in os.environ else None


def get_res_path():
    return args.res_dir


def calc_inv_density_score(x, knn):
    return np.mean(knn.kneighbors(x.reshape(1, -1))[0])


# init save directories
Path(os.path.join(get_res_path(), 'plots')).mkdir(exist_ok=True, parents=True)
Path(os.path.join(get_res_path(), 'loaders')).mkdir(exist_ok=True, parents=True)
