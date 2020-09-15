'''
loop through the candidate folder, give specific partition, give the ds nm list, draw the tsne
'''
from utils import vis
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--synset', default='ScanAva', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--gan_mode', default='lsgan', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--mode_D', default='SA', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--testset', default='Human36M', help='the gan mode [Human36M|MuPoTS]')
parser.add_argument('--proto', type=int, default=2, help='the gan mode [lsgan|vanilla]')  # proto 2
parser.add_argument('--SBL', default='n', help='if use simple baseline.')
parser.add_argument('--lmd_D', type=float, default=0.02, help='if use simple baseline.')
opt = parser.parse_args()

synset = opt.synset
testset = opt.testset
proto = opt.proto
lmd_D = opt.lmd_D
gan_mode = opt.gan_mode
mode_D = opt.mode_D

exp_li = [
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),        # C n
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode), # C y
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, gan_mode),  # for the non ds model
]

test_par = 'train'
proto = 1       # more  training human
# if_prof = True
if_prof = False
max_smpl = 100

ds_li = [
	'Human36M_p{}'.format(proto),      # different number subjs
	# 'MuCo',
	'MuPoTS',
	'ScanAva',
	'SURREAL',
]
out_fd = 'output/tsne'
if not osp.exists(out_fd):
	os.makedirs(out_fd)

for exp in exp_li:
	out_pth = osp.join(out_fd, '_'.join([exp, test_par+'.png']))     # expNm_test.png
	expNm = exp+'_{}'.format(max_smpl)
	vis.tsne_plot(osp.join('output', exp, 'vis', test_par), out_pth, ds_li=ds_li, max_num_samples_per_dataset=max_smpl, if_prof=if_prof)
