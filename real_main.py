import numpy as np
import argparse, json, pickle, math
import matplotlib.pyplot as plt
import time
import torch

import real_environments.env_ml100k, real_environments.env_wine
import algorithms

import utils
import os

"""
Define the Environment classes passed into the training algorithms
"""
ENV_CLASS = {
    'ml100k':     real_environments.env_ml100k.ml100k_Env,
    'wine':       real_environments.env_wine.wine_Env
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Configs", argument_default=argparse.SUPPRESS)
    args = utils.parse_args(parser)
    print(args.__dict__)
    
    # Run Algorithms
    ENV = ENV_CLASS[args.env](args)
    
    # ENV.reset() just sets the seed for the Env class
    ENV.reset()
    CLBBF = algorithms.CLBBF(args.T, ENV)
    CLBBF_rewards     = CLBBF.rewards()
    CLBBF_cumrewards  = np.cumsum(CLBBF_rewards)
    CLBBF_ctr         = CLBBF_cumrewards / (np.arange(args.T)+1)
    
    ENV.reset()
    OFUL = algorithms.OFUL(args.T, ENV)
    OFUL_rewards      = OFUL.rewards()
    OFUL_cumrewards   = np.cumsum(OFUL_rewards)
    OFUL_ctr          = OFUL_cumrewards / (np.arange(args.T)+1)
    
    ENV.reset()
    RANDOM = algorithms.RandomPolicy(args.T, ENV)
    RANDOM_rewards    = RANDOM.rewards()
    RANDOM_cumrewards = np.cumsum(RANDOM_rewards)
    RANDOM_ctr        = RANDOM_cumrewards / (np.arange(args.T)+1)

    # Save Results
    resultfoldertail = args.resultfoldertail
    os.makedirs(f'./real_outputs/result{resultfoldertail}/arrays', exist_ok=True)
    os.makedirs(f'./real_outputs/result{resultfoldertail}/configs', exist_ok=True) 

    timenum = round(time.time() * 1000)
    #Print out command line arguments
    arg_str = "".join('{}: {} | '.format(str(key),str(value)) for key, value in args.__dict__.items())
        #Save results in numpy format for this run
    np.savez(f'./real_outputs/result{resultfoldertail}/arrays/{args.env}_{timenum}_reward', clbbf=CLBBF_rewards, oful=OFUL_rewards,random=RANDOM_rewards)

    #Output the command line arguments to a pickle file
    with open(f'./real_outputs/result{resultfoldertail}/configs/{args.env}_{timenum}.pickle', 'wb') as handle:
        pickle.dump(args.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Output the command line arguments
    with open(f"./real_outputs/result{resultfoldertail}/{args.env}_result_guide","a+") as f:
        f.write('{}: {}'.format(timenum, arg_str))