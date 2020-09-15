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
from utils.evaluate import evaluate

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
		read all, chose onlny training partition.
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
			# regulate bb
			width = 512
			height = 512
			x, y, w, h = bbox
			x1 = np.max((0, x))
			y1 = np.max((0, y))
			x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
			y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
			if x2 >= x1 and y2 >= y1:  # don't have anno area
				bbox = np.array([x1, y1, x2 - x1, y2 - y1])
			else:
				continue
			rt_aug = 1.
			# process to ratio
			w = bbox[2] * rt_aug
			h = bbox[3] * rt_aug
			c_x = bbox[0] + w / 2.
			c_y = bbox[1] + h / 2.
			aspect_ratio = self.opts.input_shape[1] / self.opts.input_shape[0]
			if w > aspect_ratio * h:
				h = w / aspect_ratio
			elif w < aspect_ratio * h:
				w = h * aspect_ratio
			# I think there is already margin for bb, no need for 1.25?
			rt_aug = 1. # expand bb
			bbox[2] = w * rt_aug  # make bb a little larger  1.25
			bbox[3] = h * rt_aug
			bbox[0] = c_x - bbox[2] / 2.
			bbox[1] = c_y - bbox[3] / 2.

			# aggregate to entry , add to data
			entry['joint_img'] = joint_img      # x,y:pix, z:rootC-mm
			entry['joint_cam'] = joint_cam
			entry['joint_vis'] = joint_vis
			entry['img_path'] = img_path
			entry['bbox'] = bbox
			entry['root_cam'] = joint_cam[self.root_idx]
			entry['f'] = annos[i]['F']
			entry['c'] = annos[i]['C']
			data.append(entry)

		return data

	def evaluate(self, preds, **kwargs):
		'''
		rewrite this one. preds follow opts.ref_joint,  gt transfer to ref_joints, taken with ref_evals_idx. Only testset will calculate the MPJPE PA to prevent the SVD diverging during training.
		:param preds: xyz HM
		:param kwargs:  jt_adj, logger_test, if_svEval,  if_svVis
		:return:
		'''
		logger_test = kwargs.get('logger_test', None)
		if_svEval = kwargs.get('if_svEval', False)
		if_svVis = kwargs.get('if_svVis', False)
		print('Evaluation start...')
		gts = self.data
		assert (len(preds) <= len(gts))  # can be smaller
		# gts = gts[:len(preds)]  # take part of it
		# joint_num = self.joint_num
		if self.data_split == 'test':
			if_align = True
		else:
			if_align = False  # for slim evaluation

		# all can be wrapped together  ...
		# name head, ds specific
		if if_svEval:
			pth_head = '_'.join([self.opts.nmTest, self.data_split])  # ave bone only in recover not affect the HM
		else:
			pth_head = None
		# get prt func
		if logger_test:
			prt_func = logger_test.info
		else:
			prt_func = print

		# evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=self.scanNms, fn_getIdx=self.getNmIdx, opts=self.opts, if_svVis=if_svVis, pth_head=pth_head, fn_prt=prt_func)
		evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=None, fn_getIdx=None, opts=self.opts, if_svVis=if_svVis, pth_head=pth_head, fn_prt=prt_func)


if __name__ == '__main__':
	# Test case
	test_opts = {
		'ds_dir': '/scratch/liu.shu/datasets/ScanAva_1119',
		'if_tightBB_ScanAva': False,
	}
	a = ScanAva(test_opts)
