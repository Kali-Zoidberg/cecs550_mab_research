import numpy as np
import pandas as pd
import argparse, json, pickle
import matplotlib.pyplot as plt
import time
from argparse import Namespace
import glob, pickle, math, re

folders = ["_ctr_wine_20240511231615", "_ctr_wine_20240512001641"]

for resultfoldertail in folders:
    env = "wine"
    datenum_list = [ re.findall("\d+", str1)[-1] for str1 in glob.glob(f'./real_outputs/result{resultfoldertail}/configs/{env}_*.pickle')  ]
    print(datenum_list)

    CLBBF_ctrs = []
    OFUL_ctrs = []

    for datenum in datenum_list:

        load  = np.load(f'./real_outputs/result{resultfoldertail}/arrays/{env}_{datenum}_reward.npz')

        CLBBF_rewards     = load['clbbf']
        CLBBF_cumrewards  = np.cumsum(CLBBF_rewards)
        CLBBF_ctr         = CLBBF_cumrewards / (np.arange(len(CLBBF_cumrewards))+1)

        OFUL_rewards      = load['oful']
        OFUL_cumrewards   = np.cumsum(OFUL_rewards)
        OFUL_ctr          = OFUL_cumrewards / (np.arange(len(OFUL_cumrewards))+1)

        CLBBF_ctrs.append(CLBBF_ctr.copy())
        OFUL_ctrs.append(OFUL_ctr.copy())

    CLBBF_ctrs = np.vstack(CLBBF_ctrs)
    OFUL_ctrs  = np.vstack(OFUL_ctrs)

    CLBBF_avg = np.mean(CLBBF_ctrs, axis=0)
    CLBBF_std = np.std(CLBBF_ctrs, axis=0)

    OFUL_avg  = np.mean(OFUL_ctrs, axis=0)
    OFUL_std  = np.std(OFUL_ctrs, axis=0)

    T = len(CLBBF_ctr)

    indxs = np.arange(11111, T, step=11111)
    
    data = pd.DataFrame(data=np.vstack((indxs, CLBBF_avg[indxs], OFUL_avg[indxs])), index=["t", "CLBBF", "OFUL"])

    print(f"Folder: {resultfoldertail}")
    print(data)