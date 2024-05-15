import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import random

import utils
import argparse

from os import path, makedirs
from pathlib import Path

dataset_path = Path("./real_datasets/wine")
# change this variable to change the output size of the autoencoder's encoding layer
NUM_EMB_DIM = 7

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs='?', type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    X = np.vstack([np.load(dataset_path/'preprocess/X0_wine.npy'),np.load(dataset_path/'preprocess/X1_wine.npy')])
    np.random.shuffle(X)
    #Train and save model using specified num dimensions
    model = utils.AE_train(X, emb_dim=NUM_EMB_DIM, seed=seed)
    model_path = Path('./real_models/')
    makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path/f'AE_wine_s{seed}.pt')

