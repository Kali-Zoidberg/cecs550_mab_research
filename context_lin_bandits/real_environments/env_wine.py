import pandas as pd
import pickle
import math

import numpy as np
from numba import jit

import torch
import torch.nn as nn

from .utils import *


def Load_Wine(model_tail, num_partial, device):
    # assuming the program is run through the same means as the other datasets, data will be written to directory in 
    # wine_preprocess.py the same way as the other datasets.
    '''
    data_array = data.to_numpy()
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    # data (as pandas dataframes)
    # define cntext (features)
    X = wine_quality.data.features
    Y = wine_quality.data.targets

    X_split = []
    for i in range(data_array.shape[0]):
        user_id = data_array[i, 0]
        movie_id = data_array[i, 1]

        X_split.append(np.concatenate((user_id_to_feature[user_id], movie_id_to_feature[movie_id])).copy())
    X = np.vstack(X_split)
    reward0_idx = np.where(Y == 0)[0]
    reward1_idx = np.where(Y == 1)[0]
    '''
    #X0 contains samples with reward 0; X1 with reward 1
    X0 = np.load('./real_datasets/wine/preprocess/X0_wine.npy')
    X1 = np.load('./real_datasets/wine/preprocess/X1_wine.npy')

    if num_partial > 0:
        X0_len = X0.shape[0]
        X1_len = X1.shape[0]
        reward0_idxs = np.random.choice(np.arange(X0_len), num_partial, replace=False)
        reward1_idxs = np.random.choice(np.arange(X1_len), num_partial, replace=False)
        X0 = X0[reward0_idxs]
        X1 = X1[reward1_idxs]

    raw_dim = X0.shape[1]

    state_dict = torch.load('./real_models/AE_wine_{}.pt'.format(model_tail))
    emb_dim = state_dict['decoder.weight'].shape[1]

    autoencoder = Autoencoder_BN(raw_dim=raw_dim, emb_dim=emb_dim).to(device)

    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder, X0, X1, emb_dim


class wine_Env(base_Env):
    def __init__(self, args):
        super().__init__(args, Load_Wine)