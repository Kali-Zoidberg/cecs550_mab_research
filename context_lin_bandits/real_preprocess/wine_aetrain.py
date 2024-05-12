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

dataset_path = "./real_datasets/wine"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs='?', type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    X = np.vstack([np.load(dataset_path+'/preprocess/X0_wine.npy'),np.load(dataset_path+'/preprocess/X1_wine.npy')])
    np.random.shuffle(X)

    #skipping AutoEncoder for now
    model = utils.AE_train(X, seed=seed)

    torch.save(model.state_dict(), f'./real_models/AE_wine_s{seed}.pt')

