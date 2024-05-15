import subprocess, os, datetime
import numpy as np
import real_plotting

os.environ["MKL_THREADING_LAYER"] = 'GNU'
plotting_dict = {'cut':0, 'title':'Wine Dataset', 'y_lim':[0,0.13], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
seed_list = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234]
base_arg = ' --T 100000 --env wine --resultfoldertail _ctr_wine_{} '.format(now)

"""
Preprocess Data. Skip if alredy processed
"""
if os.path.exists('./real_datasets/wine/preprocess/X0_wine.npy'):
    print('Skip Preprocessing')
else: #Preprocess data by invoking the corresponding real_preprocess script for the dataset.
    print('Preprocess')
    os.system("python ./real_preprocess/wine_preprocess.py")

"""
Experiment begins here.
For each seed in the seed list, we train a new AutoEncoder to encode the dataset then call real_main to run the CMAB Algorithms
"""
for seed in seed_list:
    if os.path.exists('./real_models/AE_wine_s{}.pt'.format(seed)):
        print('Skip Training Autoencoder')
    else: #Train AutoEncoder via invoking wine_aetrain.py script
        print('Train Autoencoder')
        os.system(f"python ./real_preprocess/wine_aetrain.py --seed {seed}")

    print('Run Contextual Linear Bandit Algorithms - Seed: {}'.format(seed))
    condition_arg = f'--seed {seed} --model_tail s{str(seed)}'

    #Call real_main.py
    os.system('python ./real_main.py'+ base_arg + condition_arg)

"""
Experiments are complete by this line.
Finally, plot the values.
"""
real_plotting.plotting(resultfoldertail='_ctr_wine_{}'.format(now), env='wine', plotting_dict=plotting_dict, fnametail='_{}_all'.format(now))