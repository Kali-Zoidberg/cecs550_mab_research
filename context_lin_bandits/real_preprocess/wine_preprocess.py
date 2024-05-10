import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from ucimlrepo import fetch_ucirepo

num_cores = mp.cpu_count()
print('# of Cores: {}'.format(num_cores))

dataset_path = "./real_datasets/wine"
#assuming we don't drop any data here since there isn't anything irrelevant like user/movie ID
dataset = fetch_ucirepo(id=186)

data = pd.concat((dataset.data.features, dataset.data.targets), axis=1)

#for each sample, we set reward to 0 if quality < 5, otherwise 1
data["reward"] = np.where(data["quality"] < 5, 0, 1)
data.pop("quality")
data = data.reset_index(drop=True)

data_array = data.to_numpy()

#get the rewards as a separate vector
#dataframe is 11 attributes then the reward, so get idx 11
Y = data_array[:, 11]

X = []

reward0_idx = np.where(Y == 0)[0]
reward1_idx = np.where(Y == 1)[0]

np.save(dataset_path + '/preprocess/X0_wine', X[reward0_idx, :])
np.save(dataset_path + '/preprocess/X1_wine', X[reward1_idx, :])