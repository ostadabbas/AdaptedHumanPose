'''
tsne test
'''
import sys, os
proj_pth = os.path.join(os.path.dirname(sys.path[0]))
if proj_pth not in sys.path:
	sys.path.insert(0, proj_pth)

import numpy as np
import glob
import datetime
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.manifold import TSNE
import seaborn as sns
import os.path as osp
import utils.utils_tool as ut_t
import json
import math
import pickle
sns.set_style('whitegrid')

## check file
# fNm = 'all_joints.npy'
# fNm = 'output.npy'   # 2000 x2   float, yet not sure if this is the
# npIn = np.load(fNm, allow_pickle=True)      # 15718x3, each is object, [3 obj: jts, dsNm, index
# print(len(npIn))

dsNms = ['MuPoTS', 'Human36M', 'ScanAva', 'SURREAL']
Ns = [692, 10407, 2400, 2219]
## flow
# loop all files, get max_num_samples_per_dataset dp
# save outputs (2d fts), pths_all, steps, N_smpl)   ( inside ds indexs for later comparison

def tsne_plot(folder, max_num_samples_per_dataset, scaling=[0.5, 0.5, 0.5], exp='sa',if_shf=False):
	'''
	This version simply save everything in this folder for easy comparison.
	IF you prefer std form, another options will save result to correponding folder.
	folder: "vis/train". no shuffle version, we use even shuffling,
	output_file_path: /path/to/plot.png
	max_num_samples_per_dataset:
	scaling: scaling factors.
		example. if features are (2048, 8, 8) and scaling is (0.5, 0.75, 0.25),
		then features become (1024, 6, 2)
	exp: appended name [sa |sa-a| sa-aaic]
	fts_sv: fts_tsne_{exp}_{n_smpls}.npy        # for plot later
	'''
	output_file_path ='{}_{}.png'.format(exp, max_num_samples_per_dataset)  # auto
	print(
		'==== [{}] IMPORTANT. GENERATING TEST PLOT TO {} TO VERIFY VALID DESTINATION BEFORE GOING THROUGH COMPUTATIONS'.format(
			datetime.datetime.now(), output_file_path))
	sns.scatterplot(x=[1, 2], y=[1, 2]).get_figure().savefig(output_file_path)
	print('==== [{}] Output figure path validated. Continuing with calculations.'.format(datetime.datetime.now()))

	# datasets = os.listdir(folder)  # ['Human36M', ...]
	# Load data
	all_files = []
	labels = []

	steps = [int(math.floor(N/max_num_samples_per_dataset)*30) for N in Ns]   # for real id_num
	rgs_uni = [range(0, max_num_samples_per_dataset*step, step) for step in steps]

	print('==== [{}] Loading files from {} datasets'.format(datetime.datetime.now(), len(dsNms)))
	pths_all = []
	for i, dataset in enumerate(dsNms):

		feature_folder = os.path.join(folder, dataset, "G_fts_raw")
		if if_shf:
			print('use shuffled npys')
			numpy_files = glob.glob(os.path.join(feature_folder, "*npy"))
			np.random.shuffle(numpy_files)
			numpy_files = numpy_files[:max_num_samples_per_dataset] # shuffle get first few
		else:
			print('uniformly sampling npys')
			pth_ptn = osp.join(feature_folder, '{}.npy')
			numpy_files = [pth_ptn.format(i_smpl) for i_smpl in rgs_uni[i]]
		pths_all += numpy_files  # add current ds in
		for file in tqdm(numpy_files[:max_num_samples_per_dataset], desc=dataset):
			x = np.load(file)
			assert x.shape == (2048, 8, 8)
			all_files.append(x)
			labels.append(dataset)

	print('==== [{}] Done loading files. Loaded {} samples.'.format(datetime.datetime.now(), len(all_files)))

	# Reshape
	print('==== [{}] Downsampling features'.format(datetime.datetime.now()))
	all_files = zoom(all_files, (1,) + tuple(scaling))
	print('==== [{}] Done downsampling. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_files)))

	print('==== [{}] Reshaping feature array'.format(datetime.datetime.now()))
	new_shape = (len(all_files), np.prod(np.shape(all_files)[1:]))
	all_files = np.reshape(all_files, new_shape).astype(float)
	print('==== [{}] Done reshaping. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_files)))

	# Run t-SNE
	print('==== [{}] Running t-SNE'.format(datetime.datetime.now()))
	model = TSNE(n_components=2)
	output = model.fit_transform(all_files)
	svNm = 'fts_tsne_{}_{}.json'.format(exp, max_num_samples_per_dataset)
	svNm_mdl = 'tsne_{}_{}.pkl'.format(exp, max_num_samples_per_dataset)
	# np.save(svNm, output)
	svDict = {
		'pths_all': pths_all,     # every 500 is a ds
        'fts_tsne': output.tolist(),
		'dsNms': dsNms,
		'steps': steps,
		'N_smpl': max_num_samples_per_dataset
	}
	with open(svNm, 'w') as f:
		print('svDict type', type(svDict))
		print('f type', type(f))
		json.dump(svDict, f)
		print('tsne_fts saved at {}'.format(svNm))
	with open(svNm_mdl, 'wb') as f:
		pickle.dump(model, f)
		print("model saved at {}".format(svNm_mdl))

	# Plot
	print('==== [{}] Plotting and saving figure'.format(datetime.datetime.now()))
	snsplot = sns.scatterplot(x=output[:, 0], y=output[:, 1], hue=labels, alpha=0.7)
	snsplot.get_figure().savefig(output_file_path, dpi=300)
	print('==== [{}] Figure saved to {}.'.format(datetime.datetime.now(), output_file_path))


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--folder', default='vis/train')
	parser.add_argument('-o', '--output')
	parser.add_argument('-m', '--max-num', type=int)
	parser.add_argument('-s1', '--scaling-one', type=float, default=0.5)
	parser.add_argument('-s2', '--scaling-two', type=float, default=0.5)
	parser.add_argument('-s3', '--scaling-three', type=float, default=0.5)
	parser.add_argument('--exp', default='sa', help='the default name mainly for save')
	args = parser.parse_args()
	tsne_plot(args.folder, args.max_num, [args.scaling_one, args.scaling_two, args.scaling_three], exp=args.exp)
