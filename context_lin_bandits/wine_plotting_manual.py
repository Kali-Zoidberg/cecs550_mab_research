import subprocess, os, datetime
import numpy as np
import real_plotting


plotting_dict = {'cut':0, 'title':'Wine Dataset', 'y_lim':[0,1], 'y_ticks':[0, 0.05, 0.10], 'y_ticklabels':['0', '0.05' ,'0.10']}
now = '20240512001641'

real_plotting.plotting(resultfoldertail='_ctr_wine_{}'.format(now), env='wine', plotting_dict=plotting_dict, fnametail='_{}_all'.format(now))