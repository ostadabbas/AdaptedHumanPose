import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from utils.vis import vis_keypoints, vis_3d_skeleton
from utils.utils_pose import fliplr_joints, transform_joint_to_other_db
# from config import cfg
from math import floor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import utils_pose
from utils import utils_tool as ut_t

candidate_sets = ['Human36M', 'ScanAva', 'MPII', 'MSCOCO', 'MuCo', 'MuPoTS', 'SURREAL']

for i in range(len(candidate_sets)):
	exec('from data.' + candidate_sets[i] + '.' + candidate_sets[i] + ' import ' + candidate_sets[i])

class AdpDataset_3d(Dataset):
	'''
	adaptation datasest to make uniform interface for 3D pose. Take in all data as db original, augment, then map to original,  normalize to HM, to tensor
	changed to dict based feed e
	'''

	def __init__(self, db, ref_joints_name, is_train, transform=None, opts={}):
		self.db = db.data       # get data inherer
		self.joint_num = db.joint_num
		self.skeleton = db.skeleton
		self.flip_pairs = db.flip_pairs
		self.joints_have_depth = db.joints_have_depth
		self.if_SYN = db.if_SYN  # ds determined
		self.joints_name = db.joints_name
		self.ref_joints_name = ref_joints_name
		self.joint_num_ref = len(ref_joints_name)   # should be 17  for wht gen
		if not transform:  # if not given, give tensor normalization to overide
			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])    # maybe image needs to be normalized

		self.transform = transform
		self.is_train = is_train
		self.opts = opts

		if self.is_train:
			self.do_augment = True
		else:
			self.do_augment = False

	def __getitem__(self, index):
		joint_num = self.joint_num      # the ds jt number instead of output
		skeleton = self.skeleton
		flip_pairs = self.flip_pairs
		joints_have_depth = self.joints_have_depth
		if_SYN = self.if_SYN
		data = copy.deepcopy(self.db[index])
		bbox = data['bbox']  # give some default one for this in case not in there
		joint_img = data['joint_img']
		joint_vis = data['joint_vis']
		joint_num_ref = self.joint_num_ref
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
		img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion,
		                                        input_shape=self.opts.input_shape)  # rgb, 255 still
		for i in range(img_channels):
			img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

		# 4. generate patch joint ground truth
		# flip joints and apply Affine Transform on joints
		if do_flip:
			joint_img[:, 0] = img_width - joint_img[:, 0] - 1
			for pair in flip_pairs:
				joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :], joint_img[pair[0], :].copy()
				joint_vis[pair[0], :], joint_vis[pair[1], :] = joint_vis[pair[1], :], joint_vis[pair[0], :].copy()

		for i in range(len(joint_img)):  # 2d first for boneLen calculation
			joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)  # to pix:patch under input_shape size
		if 'joint_cam' in data.keys() and 'y' == self.opts.if_normBone:
			joint_2d = joint_img[:, 0:2]  # x,y pix:patch
			joint_cam = data['joint_cam']
			# boneLen3d = data['boneLen3d']
			boneLen2d_pix = utils_pose.get_boneLen(joint_2d, skeleton)  # rescaled bone
			boneLen2d_mm = utils_pose.get_boneLen(joint_cam[:, :2], skeleton)
			if not boneLen2d_mm>0:
				boneLen2d_mm = 3700
			rt_pix2mm = boneLen2d_pix / boneLen2d_mm

			joint_img[:, 2] *= rt_pix2mm / self.opts.input_shape[
				0]  # z = z*2d/3d /inp_size/  normalize to image size. Note, people usually use square space, yet for tight BB, sometimes it is possible depth is larger then 2d space. so I keep sensing range x:y:z = 1:1:2 ratio to include distant z. Estimation use standard z
		else:  # fix box norm
			joint_img[:, 2] /= self.opts.bbox_3d_shape[0] / 2.  # to -1,1 z total 2m

		for i in range(len(joint_img)):
			joint_img[i, 2] = (joint_img[i, 2] + 1.0) / 2.  # 0~1 normalize
			joint_vis[i] *= (       # & bitwise operation  , parenthesis ofr priority over &
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
			tmpimg = img_patch.copy().astype(np.uint8)      # rgb
			tmpkps = np.zeros((3, joint_num))
			tmpkps[:2, :] = joint_img[:, :2].transpose(1, 0)
			tmpkps[2, :] = joint_vis[:, 0]
			tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)    # rgb
			cv2.imwrite(filename + '_gt.jpg', tmpimg)

		vis = False
		if vis:
			vis_3d_skeleton(joint_img, joint_vis, skeleton, filename)
		joint_img_ori = joint_img.copy()        # raw image
		# change coordinates to output space the hmap space (usually 64 or 32 )
		joint_img[:, 0] = joint_img[:, 0] / self.opts.input_shape[1] * self.opts.output_shape[1]    # 64 x 64
		joint_img[:, 1] = joint_img[:, 1] / self.opts.input_shape[0] * self.opts.output_shape[0]
		joint_img[:, 2] = joint_img[:, 2] * self.opts.depth_dim  # 0 ~ depth_dim, from 1 to 64

		# if self.is_train: # ds completeness assumption, can feed all always
		img_patch = self.transform(img_patch)

		if self.ref_joints_name is not None:
			joint_img = transform_joint_to_other_db(joint_img, self.joints_name, self.ref_joints_name)
			joint_vis = transform_joint_to_other_db(joint_vis, self.joints_name,
			                                        self.ref_joints_name)  # invisible to no exist one

		joint_img = joint_img.astype(np.float32)  # x,y,z, hm
		joint_vis = (joint_vis > 0).astype(np.float32)
		joints_have_depth = np.array([joints_have_depth]).astype(np.float32)  # into 1 D array
		if_SYN = np.array([if_SYN]).astype(np.float32)  # bool -> float32,   to a 1D vec, to concate to N x1 (1 class)

		# make gauss_hm # n_stg:4 x n_jt:17 [sz * sz]
		# shp_t = self.opts.output_shape

		# add the evoSkel ds
		joint_evoSkel = np.zeros([16,2])  # the std joint evo , 0 to 8 10 ~ 16  16 joints
		joint_evoSkel[:9] = joint_img[:9, :2]
		joint_evoSkel[9:] = joint_img[10:, :2]
		joint_evoSkel = (joint_evoSkel - joint_evoSkel.mean(axis=0))/joint_evoSkel.std(axis=0)
		joint_evoSkel = joint_evoSkel.flatten()

		wht = np.zeros([joint_num_ref, 8, 8])   # n_jt x 8 x 8  hardwired features space
		# print(joint_img)
		for i, coord in enumerate(joint_img):     # hm all empty?
			coord = np.round(coord / 8).astype(int)
			# print(coord)
			# rst_andS = coord[0] >= 0 & coord[0] < 8 & coord[1] >= 0 & coord[1] < 8  # always false
			# print(rst_andS)
			if coord[0]>=0 and coord[0]<8 and coord[1]>=0 and coord[1]<8:
				wht[i, coord[1], coord[0]] = 1  # all 0 ???
				# print('wht {} {} {} to 1'.format(i, coord[1], coord[0]))
		wht = wht.astype(np.float32)

		# if 'n' == self.opts.if_scale_gs:        # hm gaussian,  directly cv resize to other scale directly
		# 	li2_gs = []
		# 	li_jts_gauss = []  # for jt geenration
		# 	for jt in joint_img:    # first layer
		# 		hm = np.zeros(shp_t)
		# 		hm = ut_t.draw_gaussian(hm, jt[:2], self.opts.gauss_sigma)
		# 		li_jts_gauss.append(hm)
		# 	li2_gs.append(li_jts_gauss)     # list of cv image?
		# 	sz = [shp_t[1], shp_t[0]]  # to size
		#
		# 	for i in range(1, self.opts.n_stg_D):
		# 		li_jts_t = []
		# 		sz = [e // 2 for e in sz]  # down two every time, 32, 16, 8
		# 		for hm in li_jts_gauss: # all from first layers
		# 			li_jts_t.append(transforms.ToTensor()(cv2.resize(hm, tuple(sz))).double())
		# 		li2_gs.append(li_jts_t) # add list of tensor?
		# 	for i, hm in enumerate(li_jts_gauss):
		# 		li_jts_gauss[i] = transforms.ToTensor()(hm) # to tensor
		# else:
		# 	raise ValueError('scale gaussian not implemented yet')


		# return img_patch, joint_img, joint_vis, joints_have_depth # single item
		return {'img_patch': img_patch}, {'joint_hm': joint_img, 'vis': joint_vis, 'if_depth_v': joints_have_depth,
		                                  'if_SYN_v': if_SYN, 'wts_D':wht, 'joint_img': joint_img_ori,
		                                  'joint_hm_es': joint_evoSkel}  # only image transformed to tensor other still np,  li2_gs  4x17[ 1x64x64]      # for regression later

	# else:
	# 	img_patch = self.transform(img_patch)   # optional data feed
	# 	return {'img_patch':img_patch}

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
	# return skimage RGB,  h,w,c
	img = cvimg.copy()
	img_height, img_width, img_channels = img.shape # h,w,c

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

	trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot,
	                                inv=False)  # is bb aspect needed? yes, otherwise patch distorted
	img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)  # is there channel requirements

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


def genDsLoader(opts, mode=0, if_trainShuffle=True):
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
			ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, if_trainShuffle, trans, opts=opts) # if aug here
			loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
			                    shuffle=if_trainShuffle, num_workers=opts.n_thread, pin_memory=opts.if_pinMem) # ave the batch to each ds
			itr_loader = iter(loader)       # this is purely ScanAva, why Human36M coco error?!!
			# to list
			adpDs_li.append(ds_adp)
			loader_li.append(loader)
			iter_li.append(itr_loader)
		return adpDs_li, loader_li, iter_li
	elif 1 == mode:
		ds = eval(opts.testset)("test", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts)
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)
		itr_loader = iter(loader)
		return ds_adp, loader, itr_loader
	elif 2 == mode:
		ds = eval(opts.testset)("testInLoop", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts)  # few test in loop
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)
		itr_loader = iter(loader)
		return ds_adp, loader, itr_loader
	else:
		print('no such mode defined')
		return -1


def genLoaderFromDs(ds, trans=None, opts={}, if_train=False):
	'''
	directly translate input ds input loader and adp_ds. Single ds input.
	:param ds:
	:return:
	'''
	if not trans:  # if not given, give tensor normalization to overide
		trans = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])    # no trans give default trans
	# if 'test' in ds.data_split:
	# 	is_train = False
	# else:
	# 	is_train = True
	ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, if_train, trans, opts=opts)    # false train
	loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size, shuffle=if_train, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)
	iterator = iter(loader)     # shuffle false

	return ds_adp, loader, iterator

