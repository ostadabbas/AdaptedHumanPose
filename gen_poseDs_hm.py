'''
geneate pose dataset in hm sapce.  save in  data_files fd.
Gen h36m p1 p2  and  MuCo and SURREAL
data_files <dsNm>_pm  format in json?
'''
from opt import opts, set_env, print_options
import json
import copy
from data.dataset import AdpDataset_3d
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import argparse


candidate_sets = ['Human36M', 'ScanAva', 'SURREAL', 'MPII', 'MSCOCO', 'MuCo', 'MuPoTS']
for i in range(len(candidate_sets)):
	exec('from data.' + candidate_sets[i] + '.' + candidate_sets[i] + ' import ' + candidate_sets[i])

def main():
	# ds = eval(opts.trainset[i])("train", opts=opts)
	# two opts,  list of ds  adp ds li
	# settings
	# settings
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dsNm', default='ScanAva', help='the ds to process for saving')
	parser.add_argument('--n_smpl', type=int, default=-1, help='the ds to process for saving')
	opin, _ = parser.parse_known_args()
	dsNm = opin.dsNm

	phase = 'train'
	dsNm_li = ['ScanAva', 'SURREAL', 'Human36M-p1', 'Human36M-p2', 'MuCo']
	sv_fd = 'data_files'
	n_smpl = opin.n_smpl     # how many samples to proces
	print('processing {} of n sample {}'.format(dsNm, n_smpl))

	if '1' in dsNm:
		opts.h36mProto = 1
	elif '2' in dsNm:
		opts.h36mProto = 2
	set_env(opts)       # no input_shpae why???
	# print_options(opts)
	# opts1 = opts
	# opts1.h36mProto = 1
	# opts2 = copy.copy(opts)
	# opts2.h36mProto = 2

	if not osp.exists(sv_fd):
		os.makedirs(sv_fd)
	print('opts input shape', opts.input_shape)
	# ds_li = [
	# 	ScanAva(phase, opts),
	# 	# SURREAL(phase, opts),
	# 	# Human36M(phase, opts1),
	# 	# Human36M(phase, opts2),
	# 	# MuCo(phase, opts)
	# ]
	if 'Human36M' in dsNm:
		ds = Human36M(phase, opts)
	else:
		ds = eval(dsNm)(phase, opts)

	ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, opts=opts)
	# check ds size
	print('ds len', len(ds_adp))

	# list version
	# ds_adp_li = [
	# 	AdpDataset_3d(ds, opts.ref_joints_name, False, opts=opts)
	# 	for ds in ds_li
	# ]
	# for i, ds_adp in enumerate(ds_adp_li):
	# 	jt_hm_li = []
	# 	for img, label in tqdm(ds_adp, desc='loading from {}'.format(dsNm_li[i])):
	# 		jt_hm = label['joint_hm']      # should be on cpu
	# 		vis = label['vis']      # still numpy loader auto apply  toTensor
	# 		if np.all(vis>0):       # if all visible
	# 			jt_hm_li.append(jt_hm.tolist())
	# 	sv_pth = osp.join(sv_fd, dsNm_li[i]+'_pm.json')
	# 	with open(sv_pth, 'w') as f:
	# 		json.dump(jt_hm_li, sv_pth)
	# 		print('save jt_hm to with total {} samples '.format(sv_pth, len(jt_hm_li)))

	# single ds version
	jt_hm_li = []
	cnt = 0
	for img, label in tqdm(ds_adp, desc='loading from {}'.format(dsNm)):
		if cnt== n_smpl:
			break
		jt_hm = label['joint_hm']  # should be on cpu
		vis = label['vis']  # still numpy loader auto apply  toTensor
		if np.all(vis > 0):  # if all visible
			jt_hm_li.append(jt_hm.tolist())
		cnt+=1
	sv_pth = osp.join(sv_fd, dsNm + '_hm.json')
	with open(sv_pth, 'w') as f:
		json.dump(jt_hm_li, f)
		print('save jt_hm to {} with total {} samples'.format(sv_pth, len(jt_hm_li)))
		f.close()

if __name__ == '__main__':
	main()
