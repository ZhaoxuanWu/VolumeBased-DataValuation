import os
from os.path import join as oj

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt




sigmas = [0.0001, 0.001, 0.01, 0.1, 1]
sigmas = ['1e-4', '1e-3', '1e-2', '0.1', '1']
omegas = [0.01, 0.05, 0.1, 0.2, 0.25]


# -- plot  -- 
markers = ['v', '^', '<', '>', '+', 'x']

def plot_robustness_figures(exp_dir):
    sv_txts = [ f  for f in os.listdir(exp_dir) if '.txt' in f ]
    sv_dict = {}
    for sv_txt in sv_txts:
        valuation = sv_txt.split('.txt')[0]     
        sv_dict[valuation] = np.loadtxt(oj(exp_dir, sv_txt))

    repli_times = list(range(1, 101, 5))
    plt.figure(figsize=(12, 8))

    plt.plot(repli_times, np.log(repli_times), label='$\ln$', linewidth=6)

    replicated_ig_sv = sv_dict['ig-svs']
    for row, sigma in zip(replicated_ig_sv, sigmas):
        plt.plot(repli_times, row, marker='o',label='$\\sigma$='+str(sigma), linewidth=4.5, markersize=12)


    replicated_ig_sv = sv_dict['rv-svs']
    for i, (row, omega) in enumerate(zip(replicated_ig_sv, omegas)):
        plt.plot(repli_times, row, marker=markers[i], linestyle='dashed', label='$\\omega$='+str(omega), linewidth=4.5)

    plt.legend(ncol=2, fontsize=26, loc='upper left')

    plt.ylabel("Relative Value of $\mathbf{X}_{S_1}$ ", fontsize=50)
    plt.xlabel("Replication factor $c$", fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title('IGSV vs. RVSV', fontsize=50)
    plt.tight_layout()
    plt.savefig(oj(exp_dir, 'replication-IGSV_vs_RVSV_{}.png'.format(name)))
    # plt.show()
    # exit()
    plt.clf()
    plt.close()


dirname = 'IGSV_RVSV'

for name in ['KingH','CaliH', 'USCensus', 'FaceA']:
    for t in ['disj', 'subsup']:
        exp_dir = oj(dirname, name, t)
        for spe_dir in os.listdir(exp_dir):
            plot_robustness_figures(oj(exp_dir, spe_dir))
            # exit()