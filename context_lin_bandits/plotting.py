import numpy as np
import argparse, json, pickle
import matplotlib.pyplot as plt
import time
from argparse import Namespace
import glob, pickle, math, re

import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.style.use('default')
FIGSIZE = (8,6)

def plotting(resultfoldertail, env, plotting_dict):
    
    datenum_list = [ re.findall("\d+", str1)[-1] for str1 in glob.glob(f'./result{resultfoldertail}/configs/{env}_*.pickle')  ]
    print(datenum_list)

    CLBEF_ctrs = []
    OFUL_ctrs = []
    RANDOM_ctrs = []

    for datenum in datenum_list:

        load  = np.load(f'./result{resultfoldertail}/arrays/{env}_{datenum}_reward.npz')

        CLBEF_rewards     = load['clbef']
        CLBEF_cumrewards  = np.cumsum(CLBEF_rewards)
        CLBEF_ctr         = CLBEF_cumrewards / (np.arange(len(CLBEF_cumrewards))+1)

        OFUL_rewards      = load['oful']
        OFUL_cumrewards   = np.cumsum(OFUL_rewards)
        OFUL_ctr          = OFUL_cumrewards / (np.arange(len(OFUL_cumrewards))+1)

        RANDOM_rewards    = load['random']
        RANDOM_cumrewards = np.cumsum(RANDOM_rewards)
        RANDOM_ctr        = RANDOM_cumrewards / (np.arange(len(RANDOM_cumrewards))+1)

        CLBEF_ctrs.append(CLBEF_ctr.copy())
        OFUL_ctrs.append(OFUL_ctr.copy())
        RANDOM_ctrs.append(RANDOM_ctr.copy())

    CLBEF_ctrs = np.vstack(CLBEF_ctrs)
    OFUL_ctrs  = np.vstack(OFUL_ctrs)
    RANDOM_ctrs= np.vstack(RANDOM_ctrs)

    CLBEF_avg = np.mean(CLBEF_ctrs, axis=0)
    CLBEF_std = np.std(CLBEF_ctrs, axis=0)

    OFUL_avg  = np.mean(OFUL_ctrs, axis=0)
    OFUL_std  = np.std(OFUL_ctrs, axis=0)

    RANDOM_avg = np.mean(RANDOM_ctrs, axis=0)
    RANDOM_std = np.std(RANDOM_ctrs, axis=0)
    
    ###################################################################################################
    
    T = len(CLBEF_ctr)
    
    title = plotting_dict['title']
    y_lim = plotting_dict['y_lim']
    y_ticks = plotting_dict['y_ticks']
    y_ticklabels=  plotting_dict['y_ticklabels']
    cut = plotting_dict['cut']
    legend_loc = 'lower right'

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()

    T_p = T//10
    sqrtN = float(len(datenum_list))

    x=np.array(range(T-cut))+cut
    ax.plot(x,RANDOM_avg[cut:], color='grey',label='Random',marker='s', markersize=14,markevery=T_p)
    ax.plot(x,OFUL_avg[cut:], color='lightsalmon',label='OFUL',marker='^', markersize=16,markevery=T_p)
    ax.plot(x,CLBEF_avg[cut:], color='royalblue',label='Algorithm 1',marker='o', markersize=15,markevery=T_p)
    ax.fill_between(x,RANDOM_avg[cut:] - 1.96*RANDOM_std[cut:] /sqrtN, RANDOM_avg[cut:] + 1.96*RANDOM_std[cut:] /sqrtN, color='grey', alpha =.2)
    ax.fill_between(x,OFUL_avg[cut:] - 1.96*OFUL_std[cut:] /sqrtN, OFUL_avg[cut:] + 1.96*OFUL_std[cut:] /sqrtN, color='lightsalmon', alpha =.2)
    ax.fill_between(x,CLBEF_avg[cut:] - 1.96*CLBEF_std[cut:] /sqrtN, CLBEF_avg[cut:] + 1.96*CLBEF_std[cut:] /sqrtN, color='royalblue', alpha =.2)

    ax.set_ylim(y_lim)

    ax.set_xlabel(r'Time step $t$', fontsize=25)
    ax.set_ylabel('CTR', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=22)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(22)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    ax.set_title(title,fontsize=25, pad=10)
    ax.legend(loc=legend_loc, fontsize=22)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    fig.subplots_adjust(bottom=0.17)

    fig.tight_layout()

    if cut > 0:
        cut = f'_cut{cut}'
    else:
        cut = ''

    fig.savefig(f'./result{resultfoldertail}/{env}{cut}.png',dpi=300)
    
    