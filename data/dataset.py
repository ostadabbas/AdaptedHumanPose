import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from utils.vis import vis_keypoints, vis_3d_skeleton
from utils.pose_utils import fliplr_joints, transform_joint_to_other_db
# from config import cfg
from math import floor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import pose_utils

class AdpDataset_3d(Dataset):
	'''
	adaptation datasest to make uniform interface for 3D pose
	changed to dict based feed e
	'''
	def __init__(self, db, ref_joints_name, is_train, transform, opts={}):
		self.db = db.data
		self.joint_num = db.joint_num
		self.skeleton = db.skeleton
		self.flip_pairs = db.flip_pairs
		self.joints_have_depth = db.joints_have_depth
		self.joints_name = db.joints_name
		self.ref_joints_name = ref_joints_name

		self.transform = transform
		self.is_train = is_train
		self.opts = opts

		if self.is_train:
			self.do_augment = True
		else:
			self.do_augment = False


	def __getitem__(self, index):
		joint_num = self.joint_num
		skeleton = self.skeleton
		flip_pairs = self.flip_pairs
		joints_have_depth = self.joints_have_depth

		data = copy.deepcopy(self.db[index])

		bbox = data['bbox']  # give some default one for this in case not in there
		joint_img = data['joint_img']
		joint_vis = data['joint_vis']
		# if 'boneLen' in data.keys():
		# 	boneLen = data['boneLen']   # bone length
		# else:
		# 	boneLen = 4.5       # no bone len, 2d doesn't affect at all
		# 1. load image
		cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
		if not isinstance(cvimg, np.ndarray):
			raise IOError("Fail to read %s" % data['img_path'])
		img_height, img_width, img_channels = cvimg.shape

		# 2. get augmentation params
		if self.do_augment:
			scale, rot, do_flip, color_scale, do_occlusion = get_aug_config()
		else:
			scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

		# 3. crop patch from img and perform data augmentation (flip, rot, color scale, synthetic occlusion)
		img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion, input_shape=self.opts.input_shape)
		for i in range(img_channels):
			img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

		# 4. generate patch joint ground truth
		# flip joints and apply Affine Transform on joints
		if do_flip:
			joint_img[:, 0] = img_width - joint_img[:, 0] - 1
			for pair in flip_pairs:
				joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :], joint_img[pair[0], :].copy()
				joint_vis[pair[0], :], joint_vis[pair[1], :] = joint_vis[pair[1], :], joint_vis[pair[0], :].copy()

		for i in range(len(joint_img)): # 2d first for boneLen calculation
			joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans) # to pix:patch under input_shape size
		if 'y' == self.opts.if_normBone:
			if 'joint_cam' in data.keys():
				joint_2d = joint_img[:, 0:2]
				joint_cam = data.joint_cam
				# boneLen3d = data['boneLen3d']
				boneLen2d_pix = pose_utils.get_boneLen(joint_2d, skeleton)  # rescaled bone
				boneLen2d_mm = pose_utils.get_boneLen(joint_cam[:,:2])
				rt2d_3d = boneLen2d_pix / boneLen2d_mm
				joint_img[:,2] *= rt2d_3d/self.opts.inp_sz    # z = z*2d/3d /inp_size/  normalize to image size. Note, people usually use square space, yet for tight BB, sometimes it is possible depth is larger then 2d space. so I keep x:y:z = 1:1:2 ratio to include distant z. Estimation use standard z
			else:
				print('no joint_cam info, employ 3d bb normalization instead')
				joint_img[:, 2] /= self.opts.bbox_3d_shape[0] / 2.
		else:       # fix box norm
			joint_img[:,2] /= self.opts.bbox_3d_shape[0] / 2.   # to -1,1

		for i in range(len(joint_img)):
			joint_img[i, 2] = (joint_img[i, 2] + 1.0) / 2.  # 0~1 normalize
			joint_vis[i] *= (
					(joint_img[i, 0] >= 0) & \
					(joint_img[i, 0] < self.opts.input_shape[1]) & \
					(joint_img[i, 1] >= 0) & \
					(joint_img[i, 1] < self.opts.input_shape[0]) & \
					(joint_img[i, 2] >= 0) & \
					(joint_img[i, 2] < 1)
			)  # nice filtering  all in range

		vis = False
		if vis:
			filename = str(random.randrange(1, 500))
			tmpimg = img_patch.copy().astype(np.uint8)
			tmpkps = np.zeros((3, joint_num))
			tmpkps[:2, :] = joint_img[:, :2].transpose(1, 0)
			tmpkps[2, :] = joint_vis[:, 0]
			tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
			cv2.imwrite(filename + '_gt.jpg', tmpimg)

		vis = False
		if vis:
			vis_3d_skeleton(joint_img, joint_vis, skeleton, filename)

		# change coordinates to output space the hmap space (usually 64 or 32 )
		joint_img[:, 0] = joint_img[:, 0] / self.opts.input_shape[1] * self.opts.output_shape[1]
		joint_img[:, 1] = joint_img[:, 1] / self.opts.input_shape[0] * self.opts.output_shape[0]
		joint_img[:, 2] = joint_img[:, 2] * self.opts.depth_dim

		if self.is_train:
			img_patch = self.transform(img_patch)

			if self.ref_joints_name is not None:
				joint_img = transform_joint_to_other_db(joint_img, self.joints_name, self.ref_joints_name)
				joint_vis = transform_joint_to_other_db(joint_vis, self.joints_name,
				                                        self.ref_joints_name)  # invisible to no exist one

			joint_img = joint_img.astype(np.float32)
			joint_vis = (joint_vis > 0).astype(np.float32)
			joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

			# return img_patch, joint_img, joint_vis, joints_have_depth # single item
			return {'img_patch':img_patch, 'joint_img':joint_img, 'joint_vis':joint_vis, 'joints_have_depth':joints_have_depth}

		else:
			img_patch = self.transform(img_patch)
			return {'img_patch':img_patch}


	def __len__(self):
		return len(self.db)


# helper functions
def get_aug_config():
	scale_factor = 0.25
	rot_factor = 30
	color_factor = 0.2

	scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
	rot = np.clip(np.random.randn(), -2.0,
	              2.0) * rot_factor if random.random() <= 0.6 else 0
	do_flip = random.random() <= 0.5
	c_up = 1.0 + color_factor
	c_low = 1.0 - color_factor
	color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

	do_occlusion = random.random() <= 0.5

	return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion, input_shape=(256, 256)):
	img = cvimg.copy()
	img_height, img_width, img_channels = img.shape

	# synthetic occlusion
	if do_occlusion:
		while True:
			area_min = 0.0
			area_max = 0.7
			synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

			ratio_min = 0.3
			ratio_max = 1 / 0.3
			synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

			synth_h = math.sqrt(synth_area * synth_ratio)
			synth_w = math.sqrt(synth_area / synth_ratio)
			synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
			synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

			if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
				xmin = int(synth_xmin)
				ymin = int(synth_ymin)
				w = int(synth_w)
				h = int(synth_h)
				img[ymin:ymin + h, xmin:xmin + w, :] = np.random.rand(h, w, 3) * 255
				break

	bb_c_x = float(bbox[0] + 0.5 * bbox[2])
	bb_c_y = float(bbox[1] + 0.5 * bbox[3])
	bb_width = float(bbox[2])
	bb_height = float(bbox[3])

	if do_flip:
		img = img[:, ::-1, :]
		bb_c_x = img_width - bb_c_x - 1

	trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale,
	                                rot, inv=False)
	img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)

	img_patch = img_patch[:, :, ::-1].copy()
	img_patch = img_patch.astype(np.float32)

	return img_patch, trans


def rotate_2d(pt_2d, rot_rad):
	x = pt_2d[0]
	y = pt_2d[1]
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)
	xx = x * cs - y * sn
	yy = x * sn + y * cs
	return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
	# augment size with scale
	src_w = src_width * scale
	src_h = src_height * scale
	src_center = np.array([c_x, c_y], dtype=np.float32)

	# augment rotation
	rot_rad = np.pi * rot / 180
	src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
	src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

	dst_w = dst_width
	dst_h = dst_height
	dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
	dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
	dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = src_center
	src[1, :] = src_center + src_downdir
	src[2, :] = src_center + src_rightdir

	dst = np.zeros((3, 2), dtype=np.float32)
	dst[0, :] = dst_center
	dst[1, :] = dst_center + dst_downdir
	dst[2, :] = dst_center + dst_rightdir

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans


def trans_point2d(pt_2d, trans):
	src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
	dst_pt = np.dot(trans, src_pt)
	return dst_pt[0:2]

def genDsLoader(opts, mode=0):
	'''
	generate dataset, dataLoader and iterator list (singler)
	:param opts:
	:param mode: 0 for train , 1 for test, 2 for test in loop (with folding)
	:return: datasets_li, loader_li, iterator li
	'''
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])

	if 0 == mode:
		adpDs_li, loader_li, iter_li = [], [], []

		for i in range(len(opts.trainset)):
			ds = eval(opts.trainset[i])("train", opts=opts)
			ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, True, trans, opts=opts)
			loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus*opts.batch_size//len(opts.trainset), shuffle=True, num_workers=opts.n_thread, pin_memory=True)
			iterator = iter(loader)
			# to list
			adpDs_li.append(ds_adp)
			loader_li.append(loader)
			iter_li.append(iterator)
		return adpDs_li, loader_li, iter_li

	elif 1 == mode:
		ds = eval(opts.testset)("train", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts)
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=True, num_workers=opts.n_thread, pin_memory=True)
		iterator = iter(loader)

		return ds_adp, loader, iterator
	elif 2 == mode:
		ds = eval(opts.testset)("train", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts)   # few test in loop
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=True, num_workers=opts.n_thread, pin_memory=True)
		iterator = iter(loader)

		return ds_adp, loader, iterator
	else:
		print('no such mode defined')
		return -1