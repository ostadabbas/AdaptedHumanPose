import os
import os.path as osp
import numpy as np
# from config import cfg
from utils.utils_pose import world2cam, cam2pixel, pixel2cam, rigid_align, get_bbox, warp_coord_to_ori, nameToIdx, \
	get_boneLen
import cv2
import random
import json
from utils.vis import vis_keypoints, vis_3d_skeleton
import utils.utils_pose as ut_p
import utils.utils_tool as ut_t
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path


class ScanAva:
	scanNms = [
		"SYN_RR_amir_180329_0624_G20190929_2313_P2000_A00",
"SYN_RR_behnaz_180118_2009_G20190929_2313_P2000_A00",
"SYN_RR_boya_G20191103_1634_P2000_A00",
"SYN_RR_chen_G20191103_1634_P2000_A00",
"SYN_RR_dan_jacket_180521_1312_G20190929_2322_P2000_A00",
"SYN_RR_eddy_no_coat_180517_G20190929_2332_P2000_A00",
"SYN_RR_jianchao_G20191103_1634_P2000_A00",
"SYN_RR_jinpeng_G20191103_1634_P2000_A00",
"SYN_RR_kefei_G20191103_1634_P2000_A00",
"SYN_RR_kian_180517_1605_G20190929_2342_P2000_A00",
"SYN_RR_kian_jacket_180517_1617_G20190929_2344_P2000_A00",
"SYN_RR_naveen_180403_1612_G20190929_2346_P2000_A00",
"SYN_RR_naveen_180403_1635_G20190929_2349_P2000_A00",
"SYN_RR_sarah_171201_1045_G20190929_2356_P2000_A00",
"SYN_RR_sarah_180423_1211_G20190929_2357_P2000_A00",
"SYN_RR_sarah_180423_1220_G20190930_0001_P2000_A00",
"SYN_RR_sarah_180423_1317_G20190930_0002_P2000_A00",
"SYN_RR_sharyu_G20191103_1634_P2000_A00",
"SYN_RR_shiva_G20191103_2136_P2000_A00",
"SYN_RR_shuangjun_180403_1734_G20190930_0012_P2000_A00",
"SYN_RR_shuangjun_180403_1748_G20190930_0024_P2000_A00",
"SYN_RR_shuangjun_180502_1536_G20190930_0026_P2000_A00",
"SYN_RR_shuangjun-2_G20191103_1634_P2000_A00",
"SYN_RR_shuangjun_blackT_180522_1542_G20190930_0029_P2000_A00",
"SYN_RR_shuangjun_blueSnow_180521_1531_G20190930_0029_P2000_A00",
"SYN_RR_shuangjun_diskShirt_180403_1748_G20190930_0030_P2000_A00",
"SYN_RR_shuangjun_G20191103_1634_P2000_A00",
"SYN_RR_shuangjun_grayDown_180521_1516_G20190930_0033_P2000_A00",
"SYN_RR_shuangjun_grayT_180521_1658_G20190930_0034_P2000_A00",
"SYN_RR_shuangjun_gridDshirt_180521_1548_G20190930_0035_P2000_A00",
"SYN_RR_shuangjun_jacketgood_180522_1628_G20190930_0037_P2000_A00",
"SYN_RR_shuangjun_nikeT_180522_1602_G20190930_0038_P2000_A00",
"SYN_RR_shuangjun_rainCoat_180403_1734_G20190930_0041_P2000_A00",
"SYN_RR_shuangjun_whiteDshirt_180521_1600_G20190930_0042_P2000_A00",
"SYN_RR_shuangjun_whiteHood_180502_1536_G20190930_0044_P2000_A00",
"SYN_RR_steve_2_good_color_G20190930_0045_P2000_A00",
"SYN_RR_william_180502_1449_G20190930_0049_P2000_A00",
"SYN_RR_william_180502_1509_G20190930_0050_P2000_A00",
"SYN_RR_william_180503_1704_G20190930_0059_P2000_A00",
"SYN_RR_yu_170723_1000_G20190930_0059_P2000_A00",
"SYN_RR_zishen_G20191103_1634_P2000_A00"
	]  # 1119 version  whole scan list,  train and test separation can be done in instantiation for needs. We can use last 5 for test purpose

	joint_num = 17  # for std
	joint_num_ori = 14  # truth labeled jts,
	joints_name = (
	"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
	"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
	"Neck")  # max std joints, first joint_num_ori will be true labeled
	evals_name = joints_name[:joint_num_ori + 1]  # original plus one more
	flip_pairs_name = (
		('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'),
		('R_Shoulder', 'L_Shoulder'), ('R_Elbow', 'L_Elbow'), ('R_Wrist', 'L_Wrist')
	)
	skels_name = (
		('Pelvis', 'Thorax'), ('Thorax', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		('Pelvis', 'R_Hip'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		('Pelvis', 'L_Hip'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)  # for original preferred, no Torso and Pelvis
	# boneLen2Dave_mm_cfg = {
	# 	'y': 3700,
	# 	'n': 3900
	# } # first for skelenton with different definitions, I think no need for these small differences.
	boneLen2d_av_dict = OrderedDict([('SYN_RR_william_180502_1449_G20190930_0049_P2000_A00', 3450.815146522108), ('SYN_RR_william_180503_1704_G20190930_0059_P2000_A00', 3501.008884367247), ('SYN_RR_william_180502_1509_G20190930_0050_P2000_A00', 3534.702582054287), ('SYN_RR_zishen_G20190930_0106_P2000_A00', 3406.9741603342627), ('SYN_RR_yu_170723_1000_G20190930_0059_P2000_A00', 3046.1481954101437)])
	boneLen2d_av_mm = sum(boneLen2d_av_dict.values()) / len(boneLen2d_av_dict)
	# auto generate idx
	flip_pairs = nameToIdx(flip_pairs_name, joints_name)
	skeleton = nameToIdx(skels_name, joints_name)
	eval_joint = nameToIdx(evals_name, joints_name)
	if_SYN = True
	f = (560, 560)
	c = (256, 256)      # in python this is righ pixel of center, so 255.5 should be nice

	def __init__(self, data_split, opts={}):
		self.data_split = data_split
		self.opts = opts
		self.ds_dir = opts.ds_dir
		self.img_dir = osp.join(opts.ds_dir, 'ScanAva_1119')
		self.annot_path = osp.join(opts.ds_dir, 'ScanAva_1119')
		self.human_bbox_root_dir = osp.join(opts.ds_dir, 'ScanAva_1119', 'bbox_root', 'bbox_root_scanAva_output.json')
		# self.boneLen2d_av_mm = self.boneLen2Dave_mm_cfg[opts.if_cmJoints]
		self.joints_have_depth = True
		self.root_idx = self.joints_name.index('Pelvis')
		self.lshoulder_idx = self.joints_name.index('L_Shoulder')
		self.rshoulder_idx = self.joints_name.index('R_Shoulder')
		self.rt_SYN = opts.rt_SYN
		# separation can be controlled for needs
		if self.data_split == 'train':
			self.nms_use = self.scanNms[:-5]
		elif self.data_split == 'test' or self.data_split == 'testInLoop':
			self.nms_use = self.scanNms[-5:]
		if 'y' == opts.if_tightBB_ScanAva:
			self.if_tightBB = True
		else:
			self.if_tightBB = False

		self.data = self.load_data()
		print("ScanAva {} initialized".format(data_split))

	def get_subsampling_ratio(self):
		if self.data_split == 'train':
			return self.rt_SYN
		elif self.data_split == 'test':
			return 1
		elif self.data_split == 'testInLoop':
			return 20  #
		else:
			assert 0, print('Unknown subset')

	def aug_joints(self, joint_lb):
		'''
		extend to the std joint length, then fill all missing parts. suitable for n_jt*3 format
		:param joint_lb:
		:return:
		'''
		joint_new = np.zeros((len(self.joints_name), 3))  # all 3 dim
		joint_new[:self.joint_num_ori, :] = joint_lb
		idx_lhip = self.joints_name.index('L_Hip')
		idx_rhip = self.joints_name.index('R_Hip')
		idx_pelvis = self.joints_name.index('Pelvis')
		idx_thorax = self.joints_name.index('Thorax')
		idx_head = self.joints_name.index('Head')
		idx_torso = self.joints_name.index('Torso')
		idx_neck = self.joints_name.index('Neck')

		joint_new[idx_pelvis] = (joint_new[idx_lhip] + joint_new[idx_rhip]) / 2.
		joint_new[idx_torso] = (joint_new[idx_pelvis] + joint_new[idx_thorax]) / 2.
		joint_new[idx_neck] = joint_new[idx_thorax] + 1 / 3 * (joint_new[idx_head] - joint_new[idx_thorax])
		return joint_new

	def getSubjNm(self, pth):
		'''
		from path get the subj name, this version is hard wired to search ScanAva names. Can generlaize for common tools if needed
		:param pth: is the full path of img,
		:return:
		'''
		pth = str(pth)  # to string version 11
		nm = pth.split('ScanAva_1119')[-1][1:].split('images')[0][:-1]  # get rid of last /
		return nm

	def getBoneLen_av(self, dim=2):
		boneSum_dict = OrderedDict()  # keep sum of each subject
		n_dict = OrderedDict()
		for anno in self.data:
			img_path = anno['img_path']  # eg: s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg
			joints_cam = anno['joint_cam']

			id_subjStr = self.getSubjNm(img_path) # only 2 character
			boneLen = ut_p.get_boneLen(joints_cam[:, :dim], self.skeleton)
			if id_subjStr in boneSum_dict:
				boneSum_dict[id_subjStr] += boneLen
				n_dict[id_subjStr] += 1
			else:  # first
				boneSum_dict[id_subjStr] = boneLen
				n_dict[id_subjStr] = 1

		for k in boneSum_dict:
			boneSum_dict[k] = float(boneSum_dict[k]) / n_dict[k]
		return boneSum_dict

	def load_data(self):
		'''
		from annotation file, get the sub sampled version, fill joints to std, refine path, regulate key names like C-c, F-f,
		:return:
		'''
		sampling_ratio = self.get_subsampling_ratio()  # 5 or 64 for sub sampling
		# open file
		annos = np.load(osp.join(self.annot_path, 'scanava_raw.npy'), allow_pickle=True)
		# loop all annos,
		data = []
		for i in tqdm(range(len(annos)), desc="Loading and augmenting ScanAva data..."):
			# img_path = osp.join(self.img_dir, annos[i].get('img_path').split('ScanAva_1019')[-1][1:])  # always last one with or without root path
			img_path = str(Path(self.img_dir) / Path(annos[i].get('img_path').split('ScanAva_1119/')[-1]))
			# get subj name, check in nms_use and check if in sample ratio
			nm = self.getSubjNm(img_path)
			if nm not in self.nms_use:  # no in current split
				continue
			if i % sampling_ratio != 0:  # not in sub sample
				continue

			entry = OrderedDict()
			# get original joint_img, joint_cam,
			joint_img = self.aug_joints(annos[i].get('joint_img').transpose())  # add missing points
			joint_cam = self.aug_joints(annos[i].get('joint_cam').transpose()) * 1000
			joint_cam[:, 1:] = -joint_cam[:, 1:]  # invert
			joint_vis = np.ones((len(joint_img), 1))  # no need to be int, later will multiple float
			# fill joint_img z
			joint_img[:, 2] = joint_cam[:, 2] - joint_cam[self.joints_name.index('Pelvis'), 2]

			# get bb make them regular
			if self.if_tightBB:
				bbox = ut_p.get_bbox(joint_img)
			else:  # assume the patch is square,  otherwise we need simil
				bbox = np.array([0, 0, 512, 512])
			# process to ratio
			w = bbox[2]
			h = bbox[3]
			c_x = bbox[0] + w / 2.
			c_y = bbox[1] + h / 2.
			aspect_ratio = self.opts.input_shape[1] / self.opts.input_shape[0]
			if w > aspect_ratio * h:
				h = w / aspect_ratio
			elif w < aspect_ratio * h:
				w = h * aspect_ratio
			# I think there is already margin for bb, no need for 1.25?
			rt_aug = 1.
			bbox[2] = w * rt_aug  # make bb a little larger  1.25
			bbox[3] = h * rt_aug
			bbox[0] = c_x - bbox[2] / 2.
			bbox[1] = c_y - bbox[3] / 2.

			# aggregate to entry , add to data
			entry['joint_img'] = joint_img
			entry['joint_cam'] = joint_cam
			entry['joint_vis'] = joint_vis
			entry['img_path'] = img_path
			entry['bbox'] = bbox
			entry['root_cam'] = joint_cam[self.root_idx]
			entry['f'] = annos[i]['F']
			entry['c'] = annos[i]['C']
			data.append(entry)

		return data

	def evaluate(self, preds, jt_adj=None, logger_test=None, if_svVis=False, if_svEval=False):
		'''
		rewrite this one. preds follow opts.ref_joint,  gt transfer to ref_joints, taken with ref_evals_idx.
		:param preds:
		:param jt_adj:
		:return:
		'''

		print('Evaluation start...')
		gts = self.data
		assert (len(preds) <= len(gts))  # can be smaller
		# gts = gts[:len(preds)]  # take part of it
		# joint_num = self.joint_num
		joint_num = self.opts.ref_joints_num
		sample_num = len(preds)  # use predict length
		pred_save = []
		diff_sum = np.zeros(3)  # keep x,y,z difference

		# prepare metric array
		scan_names = self.nms_use
		scan_names = [nm[7:] for nm in scan_names]  # cut starting from 7 th
		p1_error = np.zeros((len(preds), joint_num, 3))  # protocol #1 error (PA MPJPE)
		p2_error = np.zeros((len(preds), joint_num, 3))  # protocol #2 error (MPJPE)
		p1_error_action = [[] for _ in range(len(scan_names))]  # PA MPJPE for each action
		p2_error_action = [[] for _ in range(len(scan_names))]  # MPJPE error for each action

		# z metric only
		z1_error = np.zeros((len(preds), joint_num))  # protocol #1 error (PA MPJPE)
		z2_error = np.zeros((len(preds), joint_num))  # protocol #2 error (MPJPE)
		z1_error_action = [[] for _ in range(len(scan_names))]  # PA MPJPE for each action
		z2_error_action = [[] for _ in range(len(scan_names))]  # MPJPE error for each action

		for i in range(sample_num):
			gt = gts[i]
			# image_id = gt['img_id']
			f = gt['f']
			c = gt['c']
			bbox = gt['bbox']
			gt_3d_root = gt['root_cam']
			gt_3d_kpt = gt['joint_cam']
			gt_vis = gt['joint_vis']
			# file_name = gt['file_name']  # to check categories
			img_path = gt['img_path']

			if self.joints_name != self.opts.ref_joints_name:
				gt_3d_kpt = ut_p.transform_joint_to_other_db(gt_3d_kpt, self.joints_name, self.opts.ref_joints_name)
				gt_vis = ut_p.transform_joint_to_other_db(gt_vis, self.joints_name, self.opts.ref_joints_name)

			# restore coordinates to original space
			pre_2d_kpt = preds[i].copy()  # grid:Hm
			# pre_2d_kpt[:,0], pre_2d_kpt[:,1], pre_2d_kpt[:,2] = warp_coord_to_original(pre_2d_kpt, bbox, gt_3d_root)
			boneLen2d_mm = get_boneLen(gt_3d_kpt[:, :2], self.skeleton)  # individual gt bone2d_mm
			if 'y' == self.opts.if_aveBoneRec:
				boneRec = self.boneLen2d_av_mm
			else:
				boneRec = boneLen2d_mm
			pre_2d_kpt[:, 0], pre_2d_kpt[:, 1], pre_2d_kpt[:, 2] = warp_coord_to_ori(pre_2d_kpt, bbox, gt_3d_root,
			                                                                         boneLen2d_mm=boneRec,
			                                                                         opts=self.opts,
			                                                                         skel=self.opts.ref_skels_idx)  # x,y pix:cam, z mm:cam

			vis = False
			if vis:
				cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
				filename = str(random.randrange(1, 500))
				tmpimg = cvimg.copy().astype(np.uint8)
				tmpkps = np.zeros((3, joint_num))
				tmpkps[0, :], tmpkps[1, :] = pre_2d_kpt[:, 0], pre_2d_kpt[:, 1]
				tmpkps[2, :] = 1
				tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
				cv2.imwrite(filename + '_output.jpg', tmpimg)

			# back project to camera coordinate system
			pred_3d_kpt = np.zeros((joint_num, 3))
			pred_3d_kpt[:, 0], pred_3d_kpt[:, 1], pred_3d_kpt[:, 2] = pixel2cam(pre_2d_kpt, f,
			                                                                    c)  # proj back x, y, z [mm: cam]
			# adjust the pelvis position,
			if jt_adj:
				pred_3d_kpt[self.opts.ref_root_idx] += np.array(
					jt_adj)  # how much pred over gt add to get true pelvis position
			# root joint alignment
			pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.opts.ref_root_idx]  # - adj* boneLen
			gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.opts.ref_root_idx]
			# get all joints difference from gt except root
			diff_pred2gt = pred_3d_kpt - gt_3d_kpt
			diff_av = np.delete(diff_pred2gt, self.opts.ref_root_idx, axis=0).mean(
				axis=0)  # not root average           # all joints diff
			diff_sum += diff_av  # all jt diff 3
			# rigid alignment for PA MPJPE (protocol #1)
			if 'test' == self.data_split:   # only final test use the alignment to prevent the SVD converging error
				pred_3d_kpt_align = rigid_align(pred_3d_kpt, gt_3d_kpt)
			else:
				pred_3d_kpt_align = -np.ones_like(pred_3d_kpt)

			if if_svVis and 0 == i % (self.opts.svVis_step * self.opts.batch_size):  # rooted 3d
				ut_t.save_3d_tg3d(pred_3d_kpt_align, self.opts.vis_test_dir, self.opts.ref_skels_idx, idx=i,
				                  suffix='root')

			# prediction list with full joints
			pred_save.append({
				# 'img_id': image_id,  # maybe for json , use list
			                  'img_path': img_path,
			                  'joint_cam': pred_3d_kpt.tolist(),
			                  'joint_cam_aligned': pred_3d_kpt_align.tolist(),
			                  'joint_cam_gt': gt_3d_kpt.tolist(),
			                  'bbox': bbox.tolist(),
			                  'root_cam': gt_3d_root.tolist(), })

			pred_3d_kpt = np.take(pred_3d_kpt, self.opts.ref_evals_idx, axis=0)
			pred_3d_kpt_align = np.take(pred_3d_kpt_align, self.opts.ref_evals_idx, axis=0)  # take elements out to eval
			gt_3d_kpt = np.take(gt_3d_kpt, self.opts.ref_evals_idx, axis=0)

			# result and scores
			p1_error[i] = np.power(pred_3d_kpt_align - gt_3d_kpt, 2)  # PA MPJPE (protocol #1) pow2  img*jt*3
			z1_error[i] = np.abs(pred_3d_kpt_align[:, 2] - gt_3d_kpt[:, 2])  # n_jt : abs
			p2_error[i] = np.power(pred_3d_kpt - gt_3d_kpt, 2)  # MPJPE (protocol #2) n_smpl* n_jt * 3 square

			z2_error[i] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])  # n_jt

			# action_idx = int(file_name[file_name.find('act') + 4:file_name.find('act') + 6]) - 2
			scan_idx = self.nms_use.index(self.getSubjNm(img_path))   # which subj
			p1_error_action[scan_idx].append(p1_error[i].copy())
			p2_error_action[scan_idx].append(p2_error[i].copy())
			z1_error_action[scan_idx].append(z1_error[i].copy())
			z2_error_action[scan_idx].append(z2_error[i].copy())

		# reduce to metrics  into dict
		diff_av = diff_sum / sample_num
		if 'test' == self.data_split:
			p1_err_av = np.mean(np.power(np.sum(p1_error, axis=2), 0.5))  # all samp * jt
			z1_err_av = np.mean(z1_error)
		else:       # not final, use dummy one
			p1_err_av = -1
			z1_err_av = -1

		p2_err_av = np.mean(np.power(np.sum(p2_error, axis=2), 0.5))
		z2_err_av = np.mean(z2_error)
		p1_err_action_av = []
		p2_err_action_av = []
		z1_err_action_av = []
		z2_err_action_av = []

		for i in range(len(p1_error_action)):  # n_act * n_subj * n_jt * 3
			if p1_error_action[i]:  # if one array empty all empty
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
			else:       # if empty
				p1_err_action_av.append(-1)
				p2_err_action_av.append(-1)
				z1_err_action_av.append(-1)
				z2_err_action_av.append(-1)
			if 'test' == self.data_split:   #   clean up the non-final version
				p1_err_action_av[-1] = -1
				z1_err_action_av[-1] = -1

		# p1 aligned (PA MPJPE), p2 not aligned (MPJPE) , action is list
		err_dict = {
			'p1_err': p1_err_av,
			'p2_err': p2_err_av,
			'z1_err': z1_err_av,
			'z2_err': z2_err_av,
			'p1_err_act': p1_err_action_av,
			'p2_err_act': p2_err_action_av,
			'z1_err_act': z1_err_action_av,
			'z2_err_act': z2_err_action_av,
			'action_name': scan_names,
			'ref_evals_name': self.opts.ref_evals_name,  # what joints evaluated
		}

		# prepare saving
		rst_dict = {'pred_rst': pred_save, 'ref_joints_name': self.opts.ref_joints_name,
		            'jt_adj': jt_adj}  # full joints with only adj
		if if_svEval:
			rst_pth = osp.join(self.opts.rst_dir, '_'.join(['ScanAva', self.data_split,  str(len(self.opts.ref_evals_name)), self.opts.nmTest, 'rst.json']))  # Human36M_testInLoop_proto2_rst.json
			with open(rst_pth, 'w') as f:
				# json.dump(pred_save, f)
				json.dump(rst_dict, f)  # result with also meta
			print("Test result is saved at " + rst_pth)
			# save eval result
			eval_pth = osp.join(self.opts.rst_dir, '_'.join(['ScanAva', self.data_split, str(len(self.opts.ref_evals_name)),  self.opts.nmTest, 'eval.npy']))  # pickled npy
			# Human36M_testInLoop_proto2_17_eval.json
			np.save(eval_pth, err_dict)  # pickled dictionary

		# show result
		if logger_test:
			prt_func = logger_test.info
		else:
			prt_func = print
		# total
		nm_li = ['MPJPE PA', 'MPJPE', 'z PA', 'z']
		metric_li = [p1_err_av, p2_err_av, z1_err_av, z2_err_av]
		row_format = "{:>8}" + "{:>15}" * len(nm_li)
		prt_func(row_format.format("", *nm_li))
		row_format = "{:>8}" + "{:>15.1f}" * len(nm_li)
		prt_func(row_format.format("total", *metric_li))
		# action based
		row_format = "{:>8} " + "{:>15.14}" * len(scan_names)
		prt_func(row_format.format("", *scan_names))
		row_format = "{:>8}" + "{:>15.1f}" * len(scan_names)
		data_li = [p1_err_action_av, p2_err_action_av, z1_err_action_av, z2_err_action_av]
		for nm, row in zip(nm_li, data_li):
			prt_func(row_format.format(nm, *row))
		prt_func('eval diff is ' + np.array2string(diff_av))

		return err_dict


if __name__ == '__main__':
	# Test case
	test_opts = {
		'ds_dir': '/scratch/liu.shu/datasets/ScanAva_1119',
		'if_tightBB_ScanAva': False,
	}
	a = ScanAva(test_opts)
