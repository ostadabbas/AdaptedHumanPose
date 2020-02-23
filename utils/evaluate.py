'''
define the evaluate tools
'''

import numpy as np
from tqdm import tqdm
import utils.utils_tool as ut_t
import utils.utils_pose as ut_p
import cv2
import random
from utils.vis import vis_keypoints
from collections import OrderedDict
from pathlib import Path
import os.path as osp
import json


def evaluate(preds, gts, joints_name, if_align=False, act_nm_li=None, fn_getIdx=None, opts={}, avBone=3700, if_svVis=False, pth_head=None, fn_prt=print):
	'''
	Besides std metric, further include the z and diff specially for taskGEN work.
	compare preds and gts, return MPJPE,  MPJPE PA, z, z PA, diff vec
	for insufficient test (less preds than gts that some actions are missing), when missing case, the result will be filled with -1 values like PCK and MPJPE etc.
	:param preds:
	:param gts:
	:param act_nm_li: action list, this can be used for any sub categories like different subject, different scans, ...
	:param fn_getIdx:
	:param sv_head:
	:param opts:
	:param joints_name: the ds original nm_li,
	:param if_svVis: mainly for recovered 3D pose
	:return: pred_save, err_dict, pck_rst.
	pred give recovered coordinates and gts, and also image path
	err_dict:  MPJPE style evaluation
	pck_dict: MPI style evaluation
	diff_av: is the total difference from the gt, suppose to be 0 centered.
	'''

	eval_num = opts.ref_evals_num
	joint_num = opts.ref_joints_num
	jt_adj = opts.adj        # not used
	sample_num = len(preds)  # use predict length
	pred_save = []
	diff_sum = np.zeros(3)  # keep x,y,z difference

	# prepare metric array
	action_names = act_nm_li
	p1_error = np.zeros((len(preds), eval_num, 3))  # protocol #1 error (PA MPJPE)
	p2_error = np.zeros((len(preds), eval_num, 3))  # protocol #2 error (MPJPE)
	z1_error = np.zeros((len(preds), eval_num))  # protocol #1 error (PA MPJPE)
	z2_error = np.zeros((len(preds), eval_num))  # protocol #2 error (MPJPE)
	if action_names:
		p1_error_action = [[] for _ in range(len(action_names))]  # PA MPJPE for each action
		p2_error_action = [[] for _ in range(len(action_names))]  # MPJPE error for each action
		z1_error_action = [[] for _ in range(len(action_names))]  # PA MPJPE for each action
		z2_error_action = [[] for _ in range(len(action_names))]  # MPJPE error for each action

	for i in tqdm(range(sample_num), desc='evaluating ...'):
		gt = gts[i]
		f = gt['f']
		c = gt['c']
		bbox = gt['bbox']
		gt_3d_root = gt['root_cam']
		gt_3d_kpt = gt['joint_cam']
		img_path = gt['img_path']

		if joints_name != opts.ref_joints_name: #
			gt_3d_kpt = ut_p.transform_joint_to_other_db(gt_3d_kpt, joints_name, opts.ref_joints_name)
		# gt_vis = ut_p.transform_joint_to_other_db(gt_vis, self.joints_name, self.opts.ref_joints_name)

		# restore coordinates to original space
		pre_2d_kpt = preds[i].copy()  # grid:Hm
		if opts.rt_pelvisUp:
			idx_pelvis = opts.ref_joints_name.index('Pelvis')
			idx_thorax = opts.ref_joints_name.index('Thorax')
			idx_L_Hip = opts.ref_joints_name.index('L_Hip')
			idx_R_Hip = opts.ref_joints_name.index('R_Hip')
			adj_vec = opts.rt_pelvisUp * (pre_2d_kpt[idx_thorax] - pre_2d_kpt[idx_pelvis])
			pre_2d_kpt[[idx_pelvis, idx_L_Hip, idx_R_Hip]] += adj_vec
		# pre_2d_kpt[:,0], pre_2d_kpt[:,1], pre_2d_kpt[:,2] = warp_coord_to_original(pre_2d_kpt, bbox, gt_3d_root)
		boneLen2d_mm = ut_p.get_boneLen(gt_3d_kpt[:, :2], opts.ref_skels_idx)  # use standard oen
		if 'y' == opts.if_aveBoneRec:
			if not avBone:
				print('error, no aveBone given')
				exit(-1)
			else:
				boneRec = avBone
		else:
			boneRec = boneLen2d_mm
		pre_2d_kpt[:, 0], pre_2d_kpt[:, 1], pre_2d_kpt[:, 2] = ut_p.warp_coord_to_ori(pre_2d_kpt, bbox, gt_3d_root, boneLen2d_mm=boneRec, opts=opts, skel=opts.ref_skels_idx)  # x,y pix:cam, z mm:cam recover with gt/estimaed root distance

		vis = False # debug
		if vis:
			cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
			filename = str(random.randrange(1, 500))
			tmpimg = cvimg.copy().astype(np.uint8)
			tmpkps = np.zeros((3, joint_num))
			tmpkps[0, :], tmpkps[1, :] = pre_2d_kpt[:, 0], pre_2d_kpt[:, 1]
			tmpkps[2, :] = 1
			tmpimg = vis_keypoints(tmpimg, tmpkps, opts.ref_skels_idx)
			cv2.imwrite(filename + '_output.jpg', tmpimg)

		# back project to camera coordinate system
		pred_3d_kpt = np.zeros((joint_num, 3))
		pred_3d_kpt[:, 0], pred_3d_kpt[:, 1], pred_3d_kpt[:, 2] = ut_p.pixel2cam(pre_2d_kpt, f, c)  # proj back x, y, z [mm: cam]

		# adjust the pelvis position,   # abondon the adj use rt_change instead
		# if jt_adj and not opts.rt_pelvisUp:
		# 	pred_3d_kpt[opts.ref_root_idx] += np.array(
		# 		jt_adj)  # how much pred over gt add to get true pelvis position
		# root joint alignment
		pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[opts.ref_root_idx]  # - adj* boneLen
		gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[opts.ref_root_idx]
		# get all joints difference from gt except root
		diff_pred2gt = pred_3d_kpt - gt_3d_kpt
		diff_av = np.delete(diff_pred2gt, opts.ref_root_idx, axis=0).mean(
			axis=0)  # not root average           # all joints diff
		diff_sum += diff_av  # all jt diff 3
		# rigid alignment for PA MPJPE (protocol #1)

		if if_svVis and 0 == i % (opts.svVis_step * opts.batch_size):  # rooted 3d only non-aligned
			ut_t.save_3d_tg3d(pred_3d_kpt, opts.vis_test_dir, opts.ref_skels_idx, idx=i, suffix='root')

		# here take for eval
		pred_3d_kpt = np.take(pred_3d_kpt, opts.ref_evals_idx, axis=0)
		gt_3d_kpt = np.take(gt_3d_kpt, opts.ref_evals_idx, axis=0)

		# align
		if if_align:
			pred_3d_kpt_align = ut_p.rigid_align(pred_3d_kpt, gt_3d_kpt)
		else:
			pred_3d_kpt_align = -np.ones_like(pred_3d_kpt)

		# save only eval one
		pred_save.append({'img_path': img_path,
		                  'joint_cam': pred_3d_kpt.tolist(),
		                  'joint_cam_aligned': pred_3d_kpt_align.tolist(),
		                  'joint_cam_gt': gt_3d_kpt.tolist(),
		                  'bbox': bbox.tolist(),
		                  'root_cam': gt_3d_root.tolist(), })

		if_debug = False
		if if_debug:
			print('after taken')
			print('pred_3d_kpt has shape {}'.format(pred_3d_kpt.shape))
			print('pred_3d_kpt_align has shape {}'.format(pred_3d_kpt_align.shape))
			print('gt_3d_kpt has shape {}'.format(gt_3d_kpt.shape))

		# result and scores, if not align, all #1 value is fake ones.
		p1_error[i] = np.power(pred_3d_kpt_align - gt_3d_kpt, 2)  # PA MPJPE (protocol #1) pow2  img*jt*3
		z1_error[i] = np.abs(pred_3d_kpt_align[:, 2] - gt_3d_kpt[:, 2])  # n_jt : abs

		p2_error[i] = np.power(pred_3d_kpt - gt_3d_kpt, 2)  # MPJPE (protocol #2) n_smpl* n_jt * 3 square
		z2_error[i] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])  # n_jt

		if action_names:
			action_idx = fn_getIdx(img_path)
			p1_error_action[action_idx].append(p1_error[i].copy())
			p2_error_action[action_idx].append(p2_error[i].copy())
			z1_error_action[action_idx].append(z1_error[i].copy())
			z2_error_action[action_idx].append(z2_error[i].copy())

	# round up result
	if if_align:
		pck_ref = tuple(range(0, 155, 5))
		p1_err_abs = np.power(np.sum(p1_error, axis=2), 0.5)  # N x n_jt
		pck_v_tot, auc_tot = ut_t.getPCK_3d(p1_err_abs, ref=pck_ref)
		pck_v_li = []
		auc_li = []
		if action_names:
			for err_act in p1_error_action:
				if not err_act:
					pck_v = [-1] * 31
					auc = -1
				else:
					p1_act_err_abs = np.power(np.sum(err_act, axis=2), 0.5)
					pck_v, auc = ut_t.getPCK_3d(p1_act_err_abs, ref=pck_ref)
				pck_v_li.append(pck_v)
				auc_li.append(auc)

	# reduce to metrics  into dict
	diff_av = diff_sum / sample_num
	if if_align:    # p1 z1 is for aligned
		p1_err_av = np.mean(np.power(np.sum(p1_error, axis=2), 0.5))  # all samp * jt
		z1_err_av = np.mean(z1_error)
	else:  # not final, use dummy one
		p1_err_av = -1
		z1_err_av = -1

	p2_err_av = np.mean(np.power(np.sum(p2_error, axis=2), 0.5))
	z2_err_av = np.mean(z2_error)
	if action_names:
		p1_err_action_av = []
		p2_err_action_av = []
		z1_err_action_av = []
		z2_err_action_av = []

		for i in range(len(p1_error_action)):  # n_act * n_subj * n_jt * 3
			if p1_error_action[i]:  # if there is value here
				err = np.array(p1_error_action[i])
				err = np.mean(np.power(np.sum(err, axis=2), 0.5))
				p1_err_action_av.append(err)
				err = np.array(p2_error_action[i])
				err = np.mean(np.power(np.sum(err, axis=2), 0.5))
				p2_err_action_av.append(err)

				err = np.array(z1_error_action[i])
				err = np.mean(err)
				z1_err_action_av.append(err)
				err = np.array(z2_error_action[i])
				err = np.mean(err)
				z2_err_action_av.append(err)
			else:  # if empty
				p1_err_action_av.append(-1)
				p2_err_action_av.append(-1)
				z1_err_action_av.append(-1)
				z2_err_action_av.append(-1)
			if not if_align:  # clean up the non-final version
				p1_err_action_av[-1] = -1   # not align, this value
				z1_err_action_av[-1] = -1

	# p1 aligned (PA MPJPE), p2 not aligned (MPJPE) , action is list
	err_dict = {
		'p1_err': p1_err_av,
		'p2_err': p2_err_av,
		'z1_err': z1_err_av,
		'z2_err': z2_err_av,
		'ref_evals_names': opts.ref_evals_name,  # what joints evaluated
	}   # we know what joints and know what actions for the result.
	if action_names:
		err_dict['p1_err_act'] = p1_err_action_av
		err_dict['p2_err_act'] = p2_err_action_av
		err_dict['z1_err_act'] = z1_err_action_av
		err_dict['z2_err_act'] = z2_err_action_av
		err_dict['action_name'] = action_names
	# pck_rst = {'pck_v_tot':pck_v_tot, 'pck_v_li':pck_v_li, 'auc_tot':auc_tot, 'auc_li':auc_li}
	if if_align:    # pck only exists when aligned, otherwise not exists
		pck_rst = {'pck_v_tot':pck_v_tot,'auc_tot':auc_tot}
		if action_names:
			pck_rst['pck_v_li'] = pck_v_li
			pck_rst['auc_li'] = auc_li
	else:
		pck_rst = None

	if_debug= False
	if if_debug:
		pred_save0 = pred_save[0]
		for k in pred_save0:
			print('{} has type {}'.format(k, type(pred_save0[k])))
	rst_dict = {'pred_rst': pred_save, 'ref_joints_name': opts.ref_joints_name}

	# save result
	if pth_head:
		rst_dir = opts.rst_dir
		fn_prt('Test result save at {}'.format(pth_head))
		sv_json(rst_dir, pth_head, rst_dict, 'rst')
		sv_json(rst_dir, pth_head, err_dict, 'eval')
		sv_json(rst_dir, pth_head, pck_rst, '3dpck')
	prt_rst(fn_prt=fn_prt, err_dict=err_dict, pck_rst=pck_rst, diff_av=diff_av, act_nms=action_names)
	# return pred_save, err_dict, pck_rst, diff_av

def sv_json(rst_dir, pth_head, rst, sv_nm):
	pth = osp.join(rst_dir, '_'.join([pth_head, sv_nm+'.json']))
	with open(pth, 'w') as f:
		json.dump(rst, f)



def prt_rst(fn_prt=print, err_dict=None, pck_rst=None, diff_av=None, act_nms=None):
	'''
	print MPJPE style, MPI pck style, and also diff result if given
	:return:
	'''
	if err_dict:
		p1_err_av = err_dict['p1_err']
		p2_err_av = err_dict['p2_err']
		z1_err_av = err_dict['z1_err']
		z2_err_av = err_dict['z2_err']

		p1_err_act_av = err_dict.get('p1_err_act', None)
		p2_err_act_av = err_dict.get('p2_err_act', None)
		z1_err_act_av = err_dict.get('z1_err_act', None)
		z2_err_act_av = err_dict.get('z2_err_act', None)

		nm_li = ['MPJPE PA', 'MPJPE', 'z PA', 'z']
		metric_li = [p1_err_av, p2_err_av, z1_err_av, z2_err_av]
		row_format = "{:>8}" + "{:>15}" * len(nm_li)
		fn_prt(row_format.format("", *nm_li))
		row_format = "{:>8}" + "{:>15.1f}" * len(nm_li)
		fn_prt(row_format.format("total", *metric_li))
		# action based
		if p1_err_act_av:
			row_format = "{:>8}" + "{:>15.14}" * len(act_nms)
			fn_prt(row_format.format("", *act_nms))
			row_format = "{:>8}" + "{:>15.1f}" * len(act_nms)
			data_li = [p1_err_act_av, p2_err_act_av, z1_err_act_av, z2_err_act_av]
			for nm, row in zip(nm_li, data_li):
				fn_prt(row_format.format(nm, *row))

	if pck_rst:
		pck_v_tot = pck_rst.get('pck_v_tot', None)
		auc_tot = pck_rst.get('auc_tot', None)
		pck_v_li = pck_rst.get('pck_v_li', None)
		auc_li = pck_rst.get('auc_li', None)

		nm_li = ['3DPCK', 'AUC']
		metric_li = [pck_v_tot[-1], auc_tot]
		row_format = "{:>8}" + "{:>15}" * len(nm_li)
		fn_prt(row_format.format("", *nm_li))
		row_format = "{:>8}" + "{:>15.3f}" * len(nm_li)
		fn_prt(row_format.format("total", *metric_li))
		# action based
		if pck_v_li:
			row_format = "{:>8} " + "{:>15.14}" * len(act_nms)
			fn_prt(row_format.format("", *act_nms))
			row_format = "{:>8}" + "{:>15.3f}" * len(act_nms)
			pck_li = [pck[-1] for pck in pck_v_li]
			data_li = [pck_li, auc_li]
			for nm, row in zip(nm_li, data_li):
				fn_prt(row_format.format(nm, *row))

	if diff_av is not None:
		fn_prt('eval diff is' + np.array2string(diff_av))