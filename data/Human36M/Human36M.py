import os
import os.path as osp
from pycocotools.coco import COCO   # can't import here ?
import numpy as np
# from config import cfg
from utils.utils_pose import world2cam, cam2pixel, pixel2cam, rigid_align, get_bbox, warp_coord_to_ori, nameToIdx, get_boneLen
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

class Human36M:
	action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting','SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
	joint_num = 17
	joint_num_ori = 17      # truth labeled jts,
	joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Thorax', 'Neck', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist') # max std joints, first joint_num_ori will be true labeled
	evals_name = joints_name  # exact same thing
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
	boneLen2d_av_mm = sum(boneLen2d_av_dict.values()) / len(boneLen2d_av_dict)  # get average value of all bones
	# get idx
	flip_pairs = nameToIdx(flip_pairs_name, joints_name)
	skeleton = nameToIdx(skels_name, joints_name)
	eval_joint = nameToIdx(evals_name, joints_name)
	if_SYN = False

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

		# self.boneLen2d_av_mm = self.boneLen2Dave_mm_cfg[opts.if_cmJoints]
		self.joints_have_depth = True
		self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
		                    'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
		                    'WalkTogether']
		self.root_idx = self.joints_name.index('Pelvis')
		self.lshoulder_idx = self.joints_name.index('L_Shoulder')
		self.rshoulder_idx = self.joints_name.index('R_Shoulder')
		self.protocol = opts.h36mProto      # controlled outside
		self.data = self.load_data()  # load in data in preferred format
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
			# joint_cam = self.add_thorax(joint_cam)
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

		print('Evaluation start...')
		gts = self.data
		assert (len(preds) <= len(gts))     # can be smaller
		# gts = gts[:len(preds)]  # take part of it
		# joint_num = self.joint_num
		joint_num = self.opts.ref_joints_num
		sample_num = len(preds)  # use predict length
		pred_save = []
		diff_sum = np.zeros(3)      # keep x,y,z difference

		# prepare metric array
		action_name  = self.action_name
		p1_error = np.zeros((len(preds), joint_num, 3))  # protocol #1 error (PA MPJPE)
		p2_error = np.zeros((len(preds), joint_num, 3))  # protocol #2 error (MPJPE)
		p1_error_action = [[] for _ in range(len(action_name))]  # PA MPJPE for each action
		p2_error_action = [[] for _ in range(len(action_name))]  # MPJPE error for each action

		# z metric only
		z1_error = np.zeros((len(preds), joint_num))  # protocol #1 error (PA MPJPE)
		z2_error = np.zeros((len(preds), joint_num))  # protocol #2 error (MPJPE)
		z1_error_action = [[] for _ in range(len(action_name))]  # PA MPJPE for each action
		z2_error_action = [[] for _ in range(len(action_name))]  # MPJPE error for each action

		for i in tqdm(range(sample_num), desc='H36M eval...'):
			gt = gts[i]
			image_id = gt['img_id']
			f = gt['f']
			c = gt['c']
			bbox = gt['bbox']
			gt_3d_root = gt['root_cam']
			gt_3d_kpt = gt['joint_cam']
			gt_vis = gt['joint_vis']
			img_path = gt['img_path']

			if self.joints_name != self.opts.ref_joints_name:
				gt_3d_kpt = ut_p.transform_joint_to_other_db(gt_3d_kpt, self.joints_name, self.opts.ref_joints_name)
				# gt_vis = ut_p.transform_joint_to_other_db(gt_vis, self.joints_name, self.opts.ref_joints_name)

			# restore coordinates to original space
			pre_2d_kpt = preds[i].copy()  # grid:Hm
			# pre_2d_kpt[:,0], pre_2d_kpt[:,1], pre_2d_kpt[:,2] = warp_coord_to_original(pre_2d_kpt, bbox, gt_3d_root)
			boneLen2d_mm = get_boneLen(gt_3d_kpt[:, :2], self.skeleton) # individual gt
			if 'y' == self.opts.if_aveBoneRec:
				boneRec = self.boneLen2d_av_mm
			else:
				boneRec = boneLen2d_mm
			pre_2d_kpt[:, 0], pre_2d_kpt[:, 1], pre_2d_kpt[:, 2] = warp_coord_to_ori(pre_2d_kpt, bbox, gt_3d_root, boneLen2d_mm=boneRec, opts=self.opts, skel=self.opts.ref_skels_idx)  #  x,y pix:cam, z mm:cam

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
			pred_3d_kpt[:, 0], pred_3d_kpt[:, 1], pred_3d_kpt[:, 2] = pixel2cam(pre_2d_kpt, f, c)  # proj back x, y, z [mm: cam]
			# adjust the pelvis position,
			if jt_adj:
				pred_3d_kpt[self.opts.ref_root_idx] += np.array(jt_adj)  # how much pred over gt add to get true pelvis position
			# root joint alignment
			pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.opts.ref_root_idx]  # - adj* boneLen
			gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.opts.ref_root_idx]
			# get all joints difference from gt except root
			diff_pred2gt = pred_3d_kpt - gt_3d_kpt
			diff_av = np.delete(diff_pred2gt, self.opts.ref_root_idx, axis=0).mean(axis=0)  # not root average           # all joints diff
			diff_sum += diff_av     # all jt diff 3
			# rigid alignment for PA MPJPE (protocol #1)
			if 'test' == self.data_split:  # only final test use the alignment to prevent the SVD converging error
				pred_3d_kpt_align = rigid_align(pred_3d_kpt, gt_3d_kpt)
			else:
				pred_3d_kpt_align = -np.ones_like(pred_3d_kpt)

			if if_svVis and 0 == i % (self.opts.svVis_step*self.opts.batch_size):   # rooted 3d
				ut_t.save_3d_tg3d(pred_3d_kpt_align, self.opts.vis_test_dir, self.opts.ref_skels_idx, idx=i, suffix='root')

			# prediction list with full joints
			pred_save.append({'img_id': image_id,       # maybe for json , use list
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
			z2_error[i] = np.abs(pred_3d_kpt[:,2] - gt_3d_kpt[:, 2])  # n_jt

			action_idx = int(img_path[img_path.find('act') + 4:img_path.find('act') + 6]) - 2
			p1_error_action[action_idx].append(p1_error[i].copy())
			p2_error_action[action_idx].append(p2_error[i].copy())
			z1_error_action[action_idx].append(z1_error[i].copy())
			z2_error_action[action_idx].append(z2_error[i].copy())

		#reduce to metrics  into dict
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
			else:       # if empty
				p1_err_action_av.append(-1)
				p2_err_action_av.append(-1)
				z1_err_action_av.append(-1)
				z2_err_action_av.append(-1)
			if not 'test' == self.data_split:   #   clean up the non-final version
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
			'action_name': action_name,
			'ref_evals_name': self.opts.ref_evals_name,      # what joints evaluated
		}

		# prepare saving
		rst_dict = {'pred_rst': pred_save, 'ref_joints_name': self.opts.ref_joints_name, 'jt_adj': jt_adj}  # full joints with only adj
		# name head
		pth_head = '_'.join([self.opts.nmTest, self.data_split, 'proto' + str(self.protocol)])
		if if_svEval:
			rst_pth = osp.join(self.opts.rst_dir,'_'.join([pth_head, 'rst.json']))  #  Human36M_testInLoop_proto2_rst.json
			with open(rst_pth, 'w') as f:
				# json.dump(pred_save, f)
				json.dump(rst_dict, f)      # result with also meta
			print("Test result is saved at " + rst_pth)
			# save eval result
			eval_pth = osp.join(self.opts.rst_dir, '_'.join([pth_head, 'eval.npy'])) # pickled npy
			# Human36M_testInLoop_proto2_17_eval.json
			np.save(eval_pth, err_dict) # pickled dictionary

		# show result
		if logger_test:
			prt_func = logger_test.info
		else:
			prt_func = print
		# total
		prt_func('eval on {}'.format(pth_head))     # joints protocol all included
		nm_li = ['MPJPE PA', 'MPJPE', 'z PA', 'z']
		metric_li = [p1_err_av, p2_err_av, z1_err_av, z2_err_av]
		row_format = "{:>8}" + "{:>15}" * len(nm_li)
		prt_func(row_format.format("", *nm_li))
		row_format = "{:>8}" + "{:>15.1f}" * len(nm_li)
		prt_func(row_format.format("total", *metric_li))
		# action based
		row_format = "{:>8}" + "{:>15.14}" * len(action_name)
		prt_func(row_format.format("", *action_name))
		row_format = "{:>8}" + "{:>15.1f}" * len(action_name)

		data_li = [p1_err_action_av, p2_err_action_av, z1_err_action_av, z2_err_action_av]
		for nm, row in zip(nm_li, data_li):
			prt_func(row_format.format(nm, *row))
		prt_func('eval diff is' + np.array2string(diff_av))


		return err_dict