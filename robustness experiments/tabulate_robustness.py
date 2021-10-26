import os
from os.path import join as oj

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from numpy import dot
from numpy.linalg import norm


def compute_similarity(a, b):

	rho,  p = pearsonr(a, b)
	cos_sim = dot(a, b)/(norm(a)*norm(b))
	l2_norm = norm(a - b)
	l2_sim = 1 / l2_norm

	return rho, cos_sim, l2_sim 


def tabulate_robustness_svs(exp_dir):
	sv_txts = [ f  for f in os.listdir(exp_dir) if '.txt' in f ]
	sv_dict = {}
	for sv_txt in sv_txts:
		valuation = sv_txt.split('.txt')[0]		
		sv_dict[valuation] = np.loadtxt(oj(exp_dir, sv_txt))


	data_rows = []
	tl_svs = sv_dict.pop('tl-svs')
	# valuations = ['mc_tl-svs', 'tmc_tl-svs', 'ig-svs', 'v-svs', 'rv-svs-005', 'rv-svs-01']
	valuations = ['tl-loo','ig-svs', 'v-svs', 'rv-svs-005', 'rv-svs-01']
	for valuation in valuations:
		valuation_svs = sv_dict[valuation]
		average = []

		for a, b in zip(valuation_svs, tl_svs):
			similarity = compute_similarity(a, b)
			average.append(similarity)
		average = np.asarray(average).mean(axis=0)
		data_row = [valuation] + list(average)
		data_rows.append(data_row)

	df = pd.DataFrame(data_rows, columns=['Method', 'rho', 'CosSim', '1 / L2'])

	return df

name = 'USCensus'

dfs = []

exp_dir = oj(name, 'repli-against')
df = tabulate_robustness_svs(exp_dir)
dfs.append(df)

exp_dir = oj(name, 'repli-against', 'disj', '2')
df = tabulate_robustness_svs(exp_dir)
dfs.append(df)
exp_dir = oj(name, 'repli-against', 'disj', '10')
df = tabulate_robustness_svs(exp_dir)
dfs.append(df)

exp_dir = oj(name, 'repli-against', 'subsup', '1')
df = tabulate_robustness_svs(exp_dir)
dfs.append(df)

exp_dir = oj(name, 'repli-against', 'subsup', '10')
df = tabulate_robustness_svs(exp_dir)
dfs.append(df)


df = dfs[0]
# print(df.round(3))
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

for i, df_ in enumerate(dfs[1:]):
	df = df.merge(df_, on=['Method'], suffixes=['_'+ str(i+1), '_'+ str(i+2)])


result_file = oj(name, 'repli-against', 'comparison')

df.round(3).to_latex( result_file+".tex", index=False)
df.round(3).to_csv(result_file+'.csv', index=False)

# print(dfs[0].merge(dfs[1], on=['Method']).merge(dfs[2], on=['Method']).round(3))