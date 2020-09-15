import os
import os.path as osp
from pycocotools.coco import COCO   # can't import here ?
import numpy as np
# from config import cfg
from utils.utils_pose import cam2pixel, pixel2cam, rigid_align, warp_coord_to_ori, nameToIdx, get_boneLen
import cv2
import random
import json
from utils.vis import vis_keypoints, vis_3d_skeleton
import utils.utils_pose as ut_p
import utils.utils_tool as ut_t
from collections import OrderedDict
from pathlib import Path
# from Human36M_eval import calculate_score
from tqdm import tqdm
from utils.evaluate import evaluate

class Human36M:
	action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
	joint_num = 17
	joint_num_ori = 17      # truth labeled jts,
	joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Thorax', 'Neck', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist') # max std joints, first joint_num_ori will be true labeled, 11, 14 L, R shoulder
	# original name
	# self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder','L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
	# original body part torso, neck nose head,  thorax is the shoulder center.

	evals_name = joints_name  # exact same thing        # original add thorax ad the center of the shoulder , other pars are net an dnos
	flip_pairs_name = (
		('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'),
		('R_Shoulder', 'L_Shoulder'), ('R_Elbow','L_Elbow'), ('R_Wrist', 'L_Wrist')
	)
	skels_name = (
		('Pelvis', 'Torso'), ('Torso', 'Thorax'), ('Thorax', 'Neck'), ('Neck', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		('Pelvis', 'R_Hip'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		('Pelvis', 'L_Hip'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)   # for original preferred
	# boneLen2Dave_mm_cfg = {
	# 	'y': 3700,
	# 	'n': 3900
	# }
	boneLen2d_av_dict =	OrderedDict([('09', 3763.0474356854743), ('11', 3643.7693038068687)])

	boneLen2d_av_mm = sum(boneLen2d_av_dict.values()) / len(boneLen2d_av_dict)
	# get idx
	flip_pairs = nameToIdx(flip_pairs_name, joints_name)
	skeleton = nameToIdx(skels_name, joints_name)
	eval_joint = nameToIdx(evals_name, joints_name)
	if_SYN = False

	def getNmIdx(self, pth):
		action_idx = int(pth[pth.find('act') + 4:pth.find('act') + 6]) - 2
		return action_idx

	def __init__(self, data_split, opts={}):
		self.data_split = data_split
		self.opts = opts
		self.ds_dir = opts.ds_dir
		# self.img_dir = osp.join('..', 'data', 'Human36M', 'images')   # ds_dir originally is '..', 'data' with no ds_dir input
		self.img_dir = osp.join(opts.ds_dir, 'Human36M', 'images')
		self.annot_path = osp.join(opts.ds_dir, 'Human36M', 'annotations')
		self.human_bbox_root_dir = osp.join(opts.ds_dir, 'Human36M', 'bbox_root', 'bbox_root_human36m_output.json')
		# self.joint_num = 18  # repo original dataset:17, but manually added 'Thorax'
		# self.joints_name_bk = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')  # Neck -> Thorax,  Nose to Neck (here is upper neck)  , no need for the final thorax input, --repoOri

		self.joints_have_depth = True
		self.root_idx = self.joints_name.index('Pelvis')
		self.lshoulder_idx = self.joints_name.index('L_Shoulder')
		self.rshoulder_idx = self.joints_name.index('R_Shoulder')
		self.protocol = opts.h36mProto      # controlled outside
		self.data = self.load_data()  # load in data in preferred format
		if 1 == self.protocol:
			self.boneLen2d_av_mm = self.boneLen2d_av_dict['09']
		print("DS Human36M initialized")

	def get_subsampling_ratio(self):
		if self.data_split == 'train':
			return 5
		elif self.data_split == 'test':
			return 64
		elif self.data_split == 'testInLoop':
			return 640  # 10 folds of the full test
		else:
			assert 0, print('Unknown subset')

	def get_subject(self):
		if self.data_split == 'train':
			if self.protocol == 1:
				subject = [1, 5, 6, 7, 8, 9]
			elif self.protocol == 2:
				subject = [1, 5, 6, 7, 8]
		elif self.data_split == 'test' or self.data_split == 'testInLoop':
			if self.protocol == 1:
				subject = [11]
			elif self.protocol == 2:
				subject = [9, 11]
		else:
			assert 0, print("Unknown subset")

		return subject

	def getBoneLen_av(self, dim=2):
		'''
		get average bone length of each subject. Default is 2D mm length
		:param dim:
		:return:    dict of average boneLen
		'''
		boneSum_dict = OrderedDict()    # keep sum of each subject
		n_dict = OrderedDict()
		for anno in self.data:
			img_path = anno['img_path'] # eg: s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg
			joints_cam = anno['joint_cam']

			id_subjStr = img_path.split('images')[-1][3:5]  # /s_01... so from 3rd
			boneLen = ut_p.get_boneLen(joints_cam[:, :dim], self.skeleton)
			if id_subjStr in boneSum_dict:
				boneSum_dict[id_subjStr] += boneLen
				n_dict[id_subjStr] += 1
			else:   # first
				boneSum_dict[id_subjStr] = boneLen
				n_dict[id_subjStr] = 1

		for k in boneSum_dict:
			boneSum_dict[k] = float(boneSum_dict[k])/ n_dict[k]
		return boneSum_dict

	def load_data(self):
		subject_list = self.get_subject()  # 1,3,4,5,6 or 9,11
		sampling_ratio = self.get_subsampling_ratio()  # 5 or 64 for sub sampling

		# aggregate annotations from each subject
		db = COCO()
		for subject in subject_list:
			with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '.json'), 'r') as f:
				annot = json.load(f)
			if len(db.dataset) == 0:        # 1st entry
				for k, v in annot.items():
					db.dataset[k] = v  # k：images, categories,  annotations
			else:
				for k, v in annot.items():
					db.dataset[k] += v  # concat list of labels
		db.createIndex()
		# db.dataset {images,
		if 'test' in self.data_split and not self.opts.use_gt_info:  # test or testInLoop
			print("Get bounding box and root from " + self.human_bbox_root_dir)
			bbox_root_result = {}
			with open(self.human_bbox_root_dir) as f:
				annot = json.load(f)
			for i in range(len(annot)):
				bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']),
				                                               'root': np.array(annot[i]['root_cam'])}
		else:
			print("Get bounding box and root from groundtruth")

		data = []
		for aid in tqdm(db.anns.keys(), 'H36M loading'):  # loop each annotations
			ann = db.anns[aid]

			image_id = ann['image_id']
			img = db.loadImgs(image_id)[0]

			# check subject and frame_idx
			subject = img['subject'];
			frame_idx = img['frame_idx'];
			if subject not in subject_list:
				continue
			if frame_idx % sampling_ratio != 0:
				continue

			# img_path = osp.join(self.img_dir, img['file_name'])
			img_path = str(Path(self.img_dir) / img['file_name'])       # for linux/windows compatibility
			img_width, img_height = img['width'], img['height']
			cam_param = img['cam_param']
			R, t, f, c = np.array(cam_param['R']), np.array(cam_param['t']), np.array(cam_param['f']), np.array(cam_param['c'])

			# project world coordinate to cam, image coordinate space
			joint_cam = np.array(ann['keypoints_cam'])
			joint_img = np.zeros((self.joint_num, 3))
			joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = cam2pixel(joint_cam, f, c)  # in mm
			joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]  # z  root relative
			joint_vis = np.ones((self.joint_num, 1))        # 2d vis

			if 'test' in self.data_split and not self.opts.use_gt_info:
				bbox = bbox_root_result[str(image_id)][
					'bbox']  # bbox should be aspect ratio preserved-extended. It is done in RootNet.
				root_cam = bbox_root_result[str(image_id)]['root']
			else:
				bbox = np.array(ann['bbox'])
				root_cam = joint_cam[self.root_idx]

				# aspect ratio preserving bbox
				w = bbox[2]
				h = bbox[3]
				c_x = bbox[0] + w / 2.
				c_y = bbox[1] + h / 2.
				aspect_ratio = self.opts.input_shape[1] / self.opts.input_shape[0]
				if w > aspect_ratio * h:
					h = w / aspect_ratio
				elif w < aspect_ratio * h:
					w = h * aspect_ratio
				bbox[2] = w * 1.25  # make bb a little larger  1.25
				bbox[3] = h * 1.25
				bbox[0] = c_x - bbox[2] / 2.
				bbox[1] = c_y - bbox[3] / 2.

			data.append({
				'img_path': img_path,
				'img_id': image_id,
				'bbox': bbox,
				'joint_img': joint_img,  # [org_img_x, org_img_y, depth - root_depth]
				'joint_cam': joint_cam,  # [X, Y, Z] in camera coordinate
				'joint_vis': joint_vis,
				'root_cam': root_cam,  # pelvis [X, Y, Z] in camera coordinate
				'f': f,
				'c': c})
		return data

	def evaluate(self, preds, **kwargs):
		'''
		rewrite this one. preds follow opts.ref_joint,  gt transfer to ref_joints, taken with ref_evals_idx. Only testset will calculate the MPJPE PA to prevent the SVD diverging during training.
		:param preds: xyz HM
		:param kwargs:  jt_adj, logger_test, if_svEval,  if_svVis
		:return:
		'''
		jt_adj = kwargs.get('jt_adj', None)
		logger_test = kwargs.get('logger_test', None)
		if_svEval = kwargs.get('if_svEval', False)
		if_svVis  = kwargs.get('if_svVis', False)
		pth_hd = kwargs.get('pth_hd')

		print('Evaluation start...')
		gts = self.data
		assert (len(preds) <= len(gts))     # can be smaller, preds_hm!!

		if self.data_split == 'test':
			if_align = True
		else:
			if_align = False  # for slim evaluation
		if logger_test:
			prt_func = logger_test.info
		else:
			prt_func = print

		if not if_svEval:
			pth_hd = ''       #
		# must recover to the camera sapce
		evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=self.action_names, fn_getIdx=self.getNmIdx, opts=self.opts, if_svVis=if_svVis, pth_head=pth_hd, fn_prt=prt_func)

