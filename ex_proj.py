'''
project module test purpose
'''
import opt
import utils.utils_tool as ut_t
opts = opt.opts
from data.Human36M.Human36M import Human36M
from data.MPII.MPII import MPII
from utils import vis
import utils.utils_pose as ut_p
import utils.utils_tool as ut_t
import numpy as np
from skimage import io, transform
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from utils.logger import Colorlogger
from data.ScanAva.ScanAva import ScanAva
from data.MSCOCO.MSCOCO import MSCOCO
from data.MPII.MPII import MPII
from data.SURREAL.SURREAL import SURREAL
from data.MuCo.MuCo import MuCo
from data.dataset import AdpDataset_3d
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.vis import Visualizer
from models.SAA import SAA
import cv2
import torch

import os.path as osp

if __name__ == '__main__':

	## basic operation
	## test logger
	# arr1 = np.array([[2,8,6],
	#                  [9,8, 1]])
	# logger_test = Colorlogger(opts.log_dir, 'test_logs.txt')
	# logger_test.info('resut array' + np.array2string(arr1)) # can't combine string + arr, but can combine after all string

	## test ds
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])
	ds = ScanAva('train', opts)
	# ds = MSCOCO('test', opts)
	# ds = MPII('test', opts)
	# ds = SURREAL('test', opts)
	# ds = MuCo('train', opts)
	if 'test' in ds.data_split:
		is_train = False
	else:
		is_train = True
	# # get raw image and plot
	# img =cv2.imread(ds.data[0]['img_path'])
	joint_img = ds.data[0]['joint_img']
	# ut_t.showJoints(img, joint_img)

	ds_adp3d = AdpDataset_3d(ds, opts.ref_joints_name, is_train, trans, opts=opts)
	img, tar = ds_adp3d.__getitem__(15)
	# test gaussian hm
	# gauss_hm = tar['gauss_hm']
	whts = tar['whts']
	# sz correct
	# print(len(gauss_hm))
	# print(len(gauss_hm[0]))
	# print(gauss_hm[0][0].shape)
	for wht in whts:
		print(wht)

	# i_stg = 3
	# img_vis = ut_t.ts2cv2(img['img_patch'])
	# cv2.imshow('img', img_vis)
	# if_prt = 1
	# all is tensor now

	# for i in range(len(gauss_hm)):
	# 	g = gauss_hm[i][0]
	# 	print(type(g))      # suppose to be tensor now
	# 	print(g.size())
	#
	# for hm in gauss_hm[i_stg]:
	# 	if if_prt:
	# 		print(hm.shape)
	# 		if_prt = 0
	# 	cv2.imshow('gaussian hm', cv2.resize(hm.numpy().squeeze(), (256, 256))) # color not same darkness, possibly center position
	# 	cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# loader = DataLoader(dataset=ds_adp3d, batch_size=32, shuffle=True, num_workers=1)
	# nm_3d = osp.join('tmp_img3d.jpg')
	# # rg = [[0, opts.output_shape[1]], [0, opts.output_shape[0]], [0, opts.depth_dim]]
	# rg = None
	# jt_cam = ds.data[0]['joint_img']
	# print('image name', ds.data[0]['img_path'])
	# ut_t.vis_3d(jt_cam, ds.skeleton, sv_pth=nm_3d, rg=rg)    # avoid conflicts with other test in own vis_dir
	# ut_t.vis_3d(tar['joint_hm'], opts.ref_skels_idx, sv_pth=nm_3d, rg=rg)    # avoid conflicts with other test in own vis_dir
	# itr_timer = Timer()
	# itr_timer.tic()
	# itr_loader = iter(loader)       # this is purely ScanAva, why Human36M coco error?!!
	# img, tar = next(itr_loader)
	# print('get first item from iter')
	# print('iter average time is {}'.format(itr_timer.toc())) #
	#
	# itr_timer.reset()
	# itr_timer.tic()
	# for i, (img, tar) in enumerate(loader): # similar time cost as it uses iterator also
	# 	if i>0:
	# 		break
	# 	print('get item {}'.format(i))
	# itr_timer.toc()
	# print('for loop average time is {}'.format(itr_timer.toc()))
	# print(img)
	# print(tar)
	# print(img['img_patch'][0])
	# print(tar['joint_hm'][0])
	# print('job done')
	# print('dataLen')      # all these seem all right.
	# print(len(ds_scanAva.data))
	# print(ds_scanAva.data_split)
	# print('name used')
	# print(ds_scanAva.nms_use)
	# print(ds_scanAva.data[0])

	## model test
	# model = TaskGenNet(opts)
	# model.train()
	# # model.set_input({'img_patch': img}, tar)
	# model.set_input({'img_patch': img})
	# model.eval()
	# model.set_input({'img_patch': img})

	## util test
	# nm = 'checkpoint_4.pth'
	# nm_li = ['checkpoint_4.pth', 'checkpoint_2.pth']
	# cur_epoch = max([ut_t.getNumInStr(fNm)[0] for fNm in nm_li])
	# print(cur_epoch)

	## visualizer state test
	# vis = Visualizer(opts)
	# vis.plot_data = [1,2]      # class can add attribute directly
	# print(vis.plot_data)
	# print(vis.__dict__)     # all attribute even local vis item
	# rst = getattr(vis, 'haha', None)      # can't give named arg
	# print(rst)

	# skels_idx = ut_p.nameToIdx(Human36M.skels_name, Human36M.joints_name)
	# print(skels_idx)
	# jt_idx = ut_p.nameToIdx(MPII.joints_eval_name, Human36M.joints_name)
	# print(jt_idx)

	# test the resize and grid
	# img1_c6 = np.ones((6, 3,3))
	# scale_vec = np.arange(6)
	# img1_c6 = img1_c6 * scale_vec.reshape((6,1, 1))
	# fts_rsz = transform.resize(img1_c6.transpose((1, 2, 0)), (60, 60)).transpose((2, 0, 1))
	# print(fts_rsz.shape)
	# grid = ut_t.gallery(fts_rsz, ncols=3)
	# plt.imshow(grid)
	# plt.show()        # -- ok

	## ut_p test
	# ts1 = torch.arange(8).reshape(2,4)
	# ts_flip = ut_p.flip(ts1, 1)
	# print(ts_flip)

