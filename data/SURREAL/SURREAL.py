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

# official ref
# part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
#              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
#              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
#              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
#              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
#              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}
# SURREAL 3D seem x in, y up , z right  L first  .
# 2D R first, really confusing,
class SURREAL:

	joint_num = 25  # for std   put 9 to torso, double neck to head top  , neck is thorax , head-> neck,  Collar to shoulder
	joint_num_ori = 24  # truth labeled jts, first 24 is true
	joints_name = part_match = ('Pelvis', 'R_Hip', 'L_Hip',
             'Spine1', 'R_Knee','L_Knee', 'Spine2',
             'R_Ankle', 'L_Ankle', 'Torso', 'R_Foot',
             'L_Foot', 'Thorax', 'R_Collar', 'L_Collar',
             'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow',
             'L_Elbow', 'R_Wrist', 'L_Wrist', 'R_Hand', 'L_Hand', 'Head') # add head to joints
	evals_name = joints_name[:joint_num_ori + 1]  # original plus one more
	flip_pairs_name = (
		('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'), ('R_Foot', 'L_Foot'),
		('R_Collar', 'L_Collar'), ('R_Shoulder', 'L_Shoulder'), ('R_Elbow', 'L_Elbow'), ('R_Wrist', 'L_Wrist'), ('R_Hand', 'L_Hand'),
	)
	skels_name = (
		('Pelvis', 'Torso'), ('Torso', 'Thorax'), ('Thorax', 'Neck'), ('Neck', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		('Pelvis', 'R_Hip'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		('Pelvis', 'L_Hip'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)
	action_names = None
	boneLen2d_av_mm = 3590 # average over the test part, not used during train
	# auto generate idx
	flip_pairs = nameToIdx(flip_pairs_name, joints_name)
	skeleton = nameToIdx(skels_name, joints_name)
	eval_joint = nameToIdx(evals_name, joints_name)
	if_SYN = True
	f = (600, 600)
	c = (160, 120)      # in python this is righ pixel of center, so 255.5 should be nice

	getNmIdx=None       # empty func

	def __init__(self, data_split, opts={}):
		self.data_split = data_split
		self.opts = opts
		self.ds_dir = opts.ds_dir
		if opts.test_par == 'demo':
			dsNm = 'SURREAL_demo'
		else:
			dsNm = 'train_surreal_images'
		self.img_dir = osp.join(opts.ds_dir, dsNm)
		self.annot_path = osp.join(opts.ds_dir, dsNm)
		self.human_bbox_root_dir = osp.join(opts.ds_dir, dsNm, 'bbox_root', 'bbox_root_surreal_output.json')
		# self.boneLen2d_av_mm = self.boneLen2Dave_mm_cfg[opts.if_cmJoints]
		self.joints_have_depth = True
		self.root_idx = self.joints_name.index('Pelvis')
		self.lshoulder_idx = self.joints_name.index('L_Shoulder')
		self.rshoulder_idx = self.joints_name.index('R_Shoulder')
		self.rt_SYN = opts.rt_SYN
		self.train_rt = 3       # base train ratio
		self.split_rt = 0.05    # train test split

		# separation can be controlled for needs

		self.data = self.load_data()
		print("SURREAL {} initialized".format(data_split))

	def get_subsampling_ratio(self):    # total 211443
		if self.data_split == 'train':
			return self.rt_SYN*self.train_rt
		elif self.data_split == 'test' or self.data_split == 'demo':
			return 1
		elif self.data_split == 'testInLoop':
			return 10  #
		else:
			assert 0, print('Unknown subset')

	def aug_joints(self, joint_lb):
		'''
		extend to the std joint length, then fill all missing parts. suitable for n_jt*3 format
		:param joint_lb: n_jt * 3
		:return:
		'''
		joint_new = np.zeros((len(self.joints_name), 3))  # all 3 dim
		joint_new[:self.joint_num_ori, :] = joint_lb
		idx_thorax = self.joints_name.index('Thorax')
		idx_head = self.joints_name.index('Head')
		idx_neck = self.joints_name.index('Neck')

		joint_new[idx_head] = joint_new[idx_neck] + 2. * (joint_new[idx_neck] - joint_new[idx_thorax])
		return joint_new

	def getBoneLen_av(self, dim=2):
		# boneSum_dict = OrderedDict()  # keep sum of each subject
		# n_dict = OrderedDict()
		bone_sum = 0.
		N = len(self.data)
		for anno in self.data:
			img_path = anno['img_path']  # eg: s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg
			joints_cam = anno['joint_cam']
			boneLen = ut_p.get_boneLen(joints_cam[:, :dim], self.skeleton)
			bone_sum += boneLen
		bone_av = float(bone_sum)/N
		return bone_av

	def load_data(self):
		'''
		from annotation file, get the sub sampled version, fill joints to std, refine path, regulate key names like C-c, F-f.  The SURREAL 3d coord is terrible, assume camera straight point to world center.
		:return:
		'''
		sampling_ratio = self.get_subsampling_ratio()  # 5 or 64 for sub sampling
		# open file
		annos = np.load(osp.join(self.annot_path, 'surreal_annotations_raw.npy'), allow_pickle=True)
		# loop all annos,
		data = []
		N = len(annos)
		n_split = N- int(self.split_rt *N)
		if 'test' in self.data_split:
			smpl_rg = range(n_split, N)
		else:
			smpl_rg = range(n_split)
		for i in tqdm(smpl_rg, desc="Loading and augmenting SURREAL data..."):
			# img_path = osp.join(self.img_dir, annos[i].get('img_path').split('ScanAva_1019')[-1][1:])  # always last one with or without root path
			img_path = str(Path(self.img_dir) / Path(annos[i].get('image')))
			# print(img_path)
			# get subj name, check in nms_use and check if in sample ratio
			if i % sampling_ratio != 0:  # not in sub sample
				continue
			entry = OrderedDict()
			# get original joint_img, joint_cam,
			# fill joint_img 3D with joints2D with original length
			joint_img = np.zeros((self.joint_num_ori, 3))
			joint_img[:, :2] = annos[i].get('joints2D').transpose()    # n_jt * 3 filled
			joint_img = self.aug_joints(joint_img)  # extend
			# deal with 3D
			camDist = annos[i]['camDist'][0, 0]
			joint_cam = self.aug_joints(annos[i].get('joints3D').transpose()[:, ::-1] * 1000) # to mm x, y z to r, up, in
			joint_cam[:,1] = - joint_cam[:,1]       # y flip to r, d, in
			# joint_cam[:,2] = joint_cam[:,2] + camDist   # This only gives distance, however, assume cam,x,y 0, which is not correct
			joint_cam = joint_cam - np.array([0, 0, -camDist*1000])  # in their render they set height to 1000, yet x is misaligned. Yet all train and test is root centered, doesn't hurt.
			# flip 3d joints of  SURREAL wrong pairs
			for pair in self.flip_pairs:
				joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :], joint_cam[pair[0], :].copy()      # L - R exSv

			joint_vis = np.ones((len(joint_img), 1))  # just all visibile
			# fill joint_img z
			joint_img[:, 2] = joint_cam[:, 2] - joint_cam[self.joints_name.index('Pelvis'), 2]

			# get bb make them regular
			bbox = ut_p.get_bbox(joint_img)     # bb could be outside image,  then?
			width = 320
			height =240
			x,y,w, h = bbox
			x1 = np.max((0, x))
			y1 = np.max((0, y))
			x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
			y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
			if x2 >= x1 and y2 >= y1:       # don't have anno area
				bbox = np.array([x1, y1, x2 - x1, y2 - y1])
			else:
				continue
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
			entry['joint_cam'] = joint_cam      # this cam is based on root coordinate,  doesn't hurt for root base train, yet can't used for eval
			entry['joint_vis'] = joint_vis
			entry['img_path'] = img_path
			entry['bbox'] = bbox
			entry['root_cam'] = joint_cam[self.root_idx]
			entry['f'] = self.f     # use fixed version
			entry['c'] = self.c
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

		evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=self.action_names, fn_getIdx=self.getNmIdx, opts=self.opts, avBone=self.boneLen2d_av_mm, if_svVis=if_svVis, pth_head=pth_head, fn_prt=prt_func)

if __name__ == '__main__':
	# Test case
	test_opts = {
		'ds_dir': '/scratch/liu.shu/datasets/ScanAva_1119',
		'if_tightBB_ScanAva': False,
	}
	a = ScanAva(test_opts)
