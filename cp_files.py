'''
copy specific folder from A to B
'''
import os.path as osp
import os
import argparse
import json
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--synset', default='ScanAva', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--gan_mode', default='lsgan', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--mode_D', default='SA', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--testset', default='Human36M', help='the gan mode [Human36M|MuPoTS]')
parser.add_argument('--proto', type=int, default=2, help='the gan mode [lsgan|vanilla]')  # proto 2
parser.add_argument('--SBL', default='n', help='if use simple baseline.')
parser.add_argument('--lmd_D', type=float, default=0.02, help='if use simple baseline.')
opt = parser.parse_args()

testset = opt.testset
proto = opt.proto
lmd_D = opt.lmd_D
gan_mode = opt.gan_mode
mode_D = opt.mode_D
synset = opt.synset

srcFdNm = 'test'
tarFdNm = 'train'
dsNm = 'MuPoTS'     # the folder to be transfered

exp_li = [
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-C-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-SA-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-SA-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
]

for exp in tqdm(exp_li, desc='copying results...'):
	srcPth = osp.join('output', exp, 'vis', srcFdNm, dsNm)     # output/SU--/vis/test
	tarFd = osp.join('output', exp, 'vis', tarFdNm)
	if not osp.exists(tarFd):
		os.makedirs(tarFd)
	tarPth = osp.join(tarFd, dsNm)
	# shutil.copytree(srcPth, tarFd)
	shutil.copytree(srcPth, tarPth)     # from dir to gen target dir identical to this no inside