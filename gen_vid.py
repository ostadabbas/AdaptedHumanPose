'''
generate load the model and run test to generate the video
'''
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from utils.vis import vis_keypoints, genVid
from utils.utils_pose import flip
from data import dataset
import torch.backends.cudnn as cudnn
from opt import opts, print_options, set_env
import os
import os.path as osp
from utils.logger import Colorlogger
from models.SAA import SAA
import utils.utils_tool as ut_t
from utils import vis
import math
import json
from test import testLoop, weight_init

candidate_sets = ['Human36M', 'ScanAva', 'MPII', 'MSCOCO', 'MuCo', 'MuPoTS', 'SURREAL']
for i in range(len(candidate_sets)):
	exec('from data.' + candidate_sets[i] + '.' + candidate_sets[i] + ' import ' + candidate_sets[i])


def main():
	set_env(opts)
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	# cudnn.fastest = True
	cudnn.benchmark = True  #
	cudnn.deterministic = False  # result unchanged
	cudnn.enabled = True
	print('---test begins---')
	# print_options(opts)  # show options
	# test Logger here
	logger_testFinal = Colorlogger(opts.log_dir, 'testFinal_logs.txt')

	# creating models
	model = SAA(opts)  # with initialization already, already GPU-par
	logger_testFinal.info('>>> models created!! ')
	ds_test = eval(opts.testset)(opts.test_par, opts=opts)  # feet demo here
	logger_testFinal.info('>>> {} test set created'.format(opts.testset))

	# load model
	if 0 == opts.start_epoch and 'y' == opts.if_scraG:
		model.load_bb_pretrain()  # init backbone
	elif opts.start_epoch > 0:  # load the epoch model
		model.load_networks(opts.start_epoch - 1)

	testLoop(model, ds_test, opts=opts, logger_test=logger_testFinal, if_svEval=False, if_svVis=True)

	# genVideo  vis_dir with test_par
	vidSetFd = osp.join(opts.vis_dir, opts.testset)   # ds part
	if opts.testset == 'Human36M':
		vidSetFd += '_p{}'.format(opts.h36mProto)
	vidFd = osp.join(vidSetFd, 'video')
	ut_t.make_folder(vidFd)
	vis.cmb2d3d(vidSetFd)   # combine together
	fd_2d3d = osp.join(vidSetFd, '2d3d')
	vis.genVid(fd_2d3d, nm=opts.testset, svFd=vidFd)

if __name__ == '__main__':
	main()