import os
from os.path import join as oj

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_robustness_figures(exp_dir):
	sv_txts = [ f  for f in os.listdir(exp_dir) if '.txt' in f ]
	sv_dict = {}
	for sv_txt in sv_txts:
		valuation = sv_txt.split('.txt')[0]		
		sv_dict[valuation] =	np.loadtxt(oj(exp_dir, sv_txt))

	# valuations = ['tl-svs','tl-loo','ig-svs', 'v-svs', 'rv-svs-005', 'rv-svs-01']
	repli_times = list(range(1, 101, 5))
	plt.figure(figsize=(12, 8))

	plt.plot(repli_times, sv_dict['tl-loo'][:, 0], marker='^',label='LOO', linewidth=4.5)
	plt.plot(repli_times, sv_dict['tl-svs'][:, 0], marker='v',label='VLSV', linewidth=4.5)
	plt.plot(repli_times, sv_dict['ig-svs'][:, 0], marker='o',label='IGSV', linewidth=4.5)

	plt.plot(repli_times, sv_dict['v-svs'][:, 0], marker='+',label='VSV', linewidth=4.5)
	plt.plot(repli_times, sv_dict['rv-svs-01'][:, 0], marker='<',label='RVSV $\\omega$=0.1', linewidth=4.5)
	plt.plot(repli_times, sv_dict['rv-svs-005'][:, 0], marker='>',label='RVSV $\\omega$=0.05', linewidth=4.5)

	# plt.plot(repli_times, replicated_mc_tl_sv[:, 0], marker='x',label='IGSV $\\sigma$='+str(a), linewidth=3)
	# plt.plot(repli_times, replicated_tmc_tl_sv[:, 0], marker='+',label='IGSV $\\sigma$='+str(a), linewidth=3)
	plt.legend(ncol=2, fontsize=26, loc='upper left')

	plt.ylabel("Relative Value of $\mathbf{X}_{S_1}$ ", fontsize=50)
	plt.xlabel("Replication factor $c$", fontsize=50)

	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	plt.title('Valuation vs. Replication', fontsize=50)
	plt.tight_layout()
	plt.savefig(oj(exp_dir, 'replication-all_{}.png'.format(name)) )
	# plt.show()
	plt.clf()
	plt.close()

# for name in ['KingH','CaliH', 'USCensus', 'FaceA']:
	# exp_dir = oj(name, 'repli-against')
	# plot_robustness_figures(exp_dir)

for name in ['KingH','CaliH', 'USCensus', 'FaceA']:
	for t in ['disj', 'subsup']:
		exp_dir = oj(name, 'repli-against', t)
		for spe_dir in os.listdir(exp_dir):
			plot_robustness_figures(oj(exp_dir, spe_dir))
			# exit()