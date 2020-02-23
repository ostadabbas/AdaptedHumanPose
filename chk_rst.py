'''
check result, right now it is used for local PC
'''
import utils.utils_tool as ut_t
import json
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
from opt import opts
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import re


if __name__ == '__main__':
	rst_fd = r'S:\ACLab\rst_model\taskGen3d\output\ScanAva-MSCOCO-MPII_res50_n-scraG_0.1D2_y-yl_1rtSYN_regZ10_n-fG_y-nmBone_adam_lr0.001_exp'
	rst_nm = 'Human36M_y-flip_y-GtRt_jt15_avB-y_e25_exp_test_proto2_rst.json'
	ds_dir = r'S:\ACLab\datasets'
	trainLog_pth = osp.join(rst_fd, 'log', 'train_logs.txt')
	pred_name = 'joint_cam_aligned'


	with open(osp.join(rst_fd, 'result', rst_nm)) as f:
		rsts = json.load(f)['pred_rst']

	# show loss
	# # get the loss list
	# loss_li = []
	# for line in open(trainLog_pth):
	# 	m = re.search('loss:T:(\d\.\d+)', line, re.IGNORECASE)
	# 	if m:
	# 		loss_li.append(float(m.group(1)))
	# plt.plot(loss_li)
	# plt.show()

	# to vis the joint angles.
	idx_chk = 0
	for i in range(idx_chk,idx_chk+1): # check how much numbers
		rst = rsts[i]
		Pth_img = Path(rst['img_path'])
		img_pth_rel = osp.join(*Pth_img.parts[-4:])
		img = cv2.imread(osp.join(ds_dir, img_pth_rel))

		cv2.imshow('check image', img)  # checked

		# get 3d  pred, gt vis 3d in two windows, or same one
		rg = ((-1000, 1000),) *3
		joint_cam = np.array(rst[pred_name])
		joint_cam_gt = np.array(rst['joint_cam_gt'])

		# ut_t.vis_3d(joint_cam, opts.ref_skels_idx, fig_id=1, rg=rg)
		# ut_t.vis_3d(joint_cam_gt, opts.ref_skels_idx, fig_id=2, rg=rg)
		ut_t.vis_3d_cp([joint_cam, joint_cam_gt], opts.ref_skels_idx, rg=rg)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

	# get av bone and rt
	# idx_pelvis = opts.ref_joints_name.index('Pelvis')
	# idx_thorax = opts.ref_joints_name.index('Thorax')
	# torso_len_sum = 0
	# for rst in rsts:
	# 	joint_cam = np.array(rst['joint_cam'])
	# 	joint_pelvis = joint_cam[idx_pelvis]
	# 	joint_thorax = joint_cam[idx_thorax]
	# 	torso_len_sum+= joint_pelvis[1]- joint_thorax[1]    # y diff
	# torso_len_av = torso_len_sum / len(rsts)
	# rt = 35 / torso_len_av
	# print('ave torso {}, 35 to len ratio{}'.format(torso_len_av, rt))   # 453,  0.08
