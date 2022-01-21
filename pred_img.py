'''
predict single image, and generate the test image for that
'''
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from utils.vis import vis_keypoints
from utils.utils_pose import flip
from data import dataset
import torch.backends.cudnn as cudnn
from opt import opts, print_options
import os
import os.path as osp
from utils.logger import Colorlogger
from models.SAA import SAA
from data.dataset import genLoaderFromDs
import utils.utils_tool as ut_t
import math
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('tkagg')

if __name__=='__main__':
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	cudnn.fastest = True
	cudnn.benchmark = True
	cudnn.deterministic = False  # result unchanged
	cudnn.enabled = True
	print('---test begins---')
	print_options(opts)  # show options

	# get image in
	img_pth = Path('h36m_samples/s_09_act_02_subact_01_ca_01_000001_crop.jpg')
	img = cv2.imread(str(img_pth))
	img = cv2.resize(img, (opts.inp_sz, opts.inp_sz))   # resize
	img_sk = img[:,:,::-1].astype(np.float32)  #  to skimage float
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])
	img_ts = trans(img_sk)     # to tensor normalized
	input = {'img_patch': img_ts.unsqueeze(0)}

	# model
	model = SAA(opts)  # with initialization already, already GPU-par
	if 0 == opts.start_epoch and 'y' == opts.if_scraG:
		model.load_bb_pretrain()  # init backbone
	elif opts.start_epoch > 0:  # load the epoch model
		model.load_networks(opts.start_epoch - 1)
	with torch.no_grad():
		model.set_input(input)
		model.forward()
		coord_out = model.coord.cpu().numpy()

	pred2d_patch = np.zeros((3, opts.ref_joints_num))  # 3xn_jt format
	pred2d_patch[:2, :] = coord_out[0, :, :2].transpose(1, 0) / opts.output_shape[0] * opts.input_shape[0]  # 3 * n_jt  set depth 1 , from coord_out 0 !!
	pred2d_patch[2, :] = 1
	vis2d = vis_keypoints(img, pred2d_patch, opts.ref_skels_idx)
	cv2.imshow('2d img', vis2d)     # only show
	rg_hm = ((0,64), )*3
	ut_t.vis_3d(coord_out[0], opts.ref_skels_idx, rg=rg_hm)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
