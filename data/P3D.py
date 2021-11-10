'''
the pose 3D loader
'''
from torch.utils.data import Dataset
import os.path as osp
import os
import json
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import torch
import utils.utils_pose as ut_p
from data.Human36M.Human36M import Human36M

idx_thorax = 8  #

class P3D_D(Dataset):
	def __init__(self, preds, gts, opts=None, split='train', deg_rng=[30, 180, 30], scal_rng=[0.8, 1.2]):
		'''
		the D version feeding, get src and tar json path. feed back with pred_tch, gt_tch, skel_pred, skel_tar,  pred , gt ( matched or not matched) ,
		skeletons
		:param opts:
		:param split:
		:param deg_rng: the rotation range of transformation.
		:param scal_rng: the scaling range
		'''
		self.deg_rng = deg_rng
		self.scal_rng = scal_rng
		self.preds = preds  # pred gt of the h36m trainset
		self.gts = gts      #   0 ~ 64 space
		self.if_neckRt = opts.if_neckRt
		self.split = split
		self.if_aug = False
		self.preds_thrx = preds[:, (idx_thorax,)]
		self.gts_thrx = gts[:, (idx_thorax,)]
		self.joints_name = Human36M.joints_name
		self.skel_norm_pair = ut_p.nameToIdx(['L_Shoulder', 'R_Shoulder'], self.joints_name)
		self.skeleton = Human36M.skeleton
		self.if_skel_av = opts.if_skel_av

		li_skel_inp = []
		for jt in self.preds:
			skelT = ut_p.get_l_skels(jt, self.skeleton, self.skel_norm_pair)        # implement,  torch and also numpy version
			li_skel_inp.append(skelT)
		self.l_skel_inp = np.array(li_skel_inp).astype(np.float32)
		self.l_skel_inp_av = self.l_skel_inp.mean(axis=0)   # 17 x3

		li_skel_tar = []
		for jt in self.gts:
			skelT = ut_p.get_l_skels(jt, self.skeleton, self.skel_norm_pair)
			li_skel_tar.append(skelT)
		self.l_skel_tar = np.array(li_skel_tar).astype(np.float32)
		self.l_skel_tar_av = self.l_skel_tar.mean(axis=0)

		# if split == 'train' and opts.if_VPR == 'y':  # if train and aug  set true
		# 	self.if_aug = True
		# 	print('>>> using VPR')

		if split=='test':
			# self.if_aug = False
			assert len(self.preds) == len(self.gts), 'pred {} and gt {} should have same length'.format(len(self.preds),len(self.gts))

		self.N = len(self.preds)
		self.N_gt = len(self.gts)   # for random choose

	def __getitem__(self, idx):
		if self.split == 'train':
			idx_B = np.random.randint(0, self.N_gt - 1)
		else:
			idx_B = idx
		pred = self.preds[idx]  # use std version
		gt = self.gts[idx_B]        # random or the same


		if self.if_neckRt == 'y':       # process during feeding
			pred = pred - pred[idx_thorax]
			gt = gt - gt[idx_thorax]

		if self.if_aug:     # always aug control here
			deg_rng = self.deg_rng
			scal_rng = self.scal_rng
			deg_x = random.uniform(-deg_rng[0], deg_rng[0])
			deg_y = random.uniform(-deg_rng[1], deg_rng[1])
			deg_z = random.uniform(-deg_rng[2], deg_rng[2])
			scal = random.uniform(scal_rng[0], scal_rng[1])

			# rtm = R.from_euler('xyz', angles=[deg_x, deg_y, deg_z])     # other rotation doesn't help only y?
			# rtm = R.from_euler('xyz', angles=[0, deg_y, 0])     # only rotate y
			# rtm = R.from_euler('xyz', angles=[deg_x,0,  0])     # only rotate y
			rtm = R.from_euler('xyz', angles=[0, 0, 0])  # only scale
			# print('matrix is', rtm.as_matrix())
			# set scale to 1 if no aug
			# scale= 1.
			pred = rtm.apply(pred * scal)  # 17 x 3  new coordinate
			gt = rtm.apply(gt * scal)  # to new scale and transformation

		pred_tch = torch.from_numpy(pred.flatten()).float()  # to one dim
		gt_tch = torch.from_numpy(gt.flatten()).float()  # to one dim
		# skel_pred = torch.from_numpy(l_skels_pred.astype(np.float32))
		# skel_gt = torch.from_numpy(l_skels_gt.astype(np.float32))
		if self.if_skel_av == 'n':
			skel_pred = torch.from_numpy(self.l_skel_inp[idx])
			skel_tar = torch.from_numpy(self.l_skel_tar[idx_B])
		else:       # gives average skels
			skel_pred = torch.from_numpy(self.l_skel_inp_av)
			skel_tar = torch.from_numpy(self.l_skel_tar_av)

		# return src and tar  , flattened
		return pred_tch, gt_tch, skel_pred, skel_tar

	def __len__(self):
		return self.N

class P3D(Dataset):
	def __init__(self, opts, split='train', deg_rng =[30, 180, 30], scal_rng = [0.8, 1.2]):
		'''
		according to split to select the set. after recover, the size could vary from each other.
		:param opts:
		:param split:
		:param deg_rng: the rotation range of transformation.
		:param scal_rng: the scaling range
		'''
		tarset = opts.tarset_PA
		self.deg_rng = deg_rng
		self.scal_rng = scal_rng
		if opts.if_norm_PA == 'y':
			self.if_norm = True
		else:
			self.if_norm = False

		if opts.if_hm_PA:
			str_cfrm = 'hm'     # coordinate frame,
		else:
			str_cfrm = 'camRt'

		if tarset=='MuCo' and split=='test':        # use the MuPoTS instead  here
			# 'Human36M_Btype-h36m_SBL-n_PA-n_exp_train_proto2_pred_hm.npy'
			dsNm = 'MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test_pred_{}'.format(str_cfrm)      # PA-n can be followed by the VRT if given
			# dt_pth = osp.join(opts.rst_dir, dsNm)
		else:
			if tarset == 'h36m-p1':
				dsNm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_{}_proto1_pred_{}'.format(split, str_cfrm)
			elif tarset == 'h36m-p2':
				dsNm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_{}_proto2_pred_{}'.format(split, str_cfrm)
			elif tarset =='MuPoTS':
				dsNm = 'MuCo_Btype-h36m_SBL-n_PA-n_exp_{}_pred_{}'.format(split, str_cfrm)        # only train

		dt_pth = osp.join(opts.rst_dir, dsNm+'.json')

		print('>>>loading from {}'.format(dt_pth))
		with open(dt_pth, 'r') as f:
			dtIn = json.load(f)
			f.close()
		self.preds = np.array(dtIn['pred'])     # pred gt of the h36m trainset
		self.gts = np.array(dtIn['gt'])
		assert len(self.preds) == len(self.gts), 'pred {} and gt {} should have same length'.format(len(self.preds), len(self.gts))

		if split == 'train' and opts.if_VPR == 'y':  # if train and aug  set true
			self.if_aug = True
			print('>>> using VPR')
		else:
			self.if_aug = False
		self.N = len(self.preds)
		# get std
		stat_pth = osp.join(opts.rst_dir, dsNm + '_stat.json')
		if not osp.exists(stat_pth):
			print('>>>Failed to found std.json, recreating...')
			mu_pred = self.preds.mean(axis=0)     # 17 x3
			std_pred = self.preds.std(axis=0)
			mu_gt = self.gts.mean(axis=0)
			std_gt = self.gts.std(axis=0)
			stat = {'mu_pred': mu_pred.tolist(),
			        'std_pred':std_pred.tolist(),
			        'mu_gt': mu_gt.tolist(),
			        'std_gt':std_gt.tolist()}
			with open(stat_pth, 'w') as f:
				json.dump(stat, f)
				f.close()

		with open(stat_pth, 'r') as f:
			statIn = json.load(f)
			self.mu_pred = np.array(statIn['mu_pred'])
			self.std_pred = np.array(statIn['std_pred'])
			self.mu_gt = np.array(statIn['mu_pred'])
			self.std_gt = np.array(statIn['std_pred'])

		# make pelvis std to be 1
		self.std_pred[0,:] = 1
		self.std_gt[0,:] = 1

		self.preds_nmd = (self.preds - self.mu_pred)/self.std_pred
		self.gts_nmd = (self.gts - self.mu_gt)/self.std_gt


	def __getitem__(self, idx):
		if self.if_norm:
			pred = self.preds_nmd[idx]  # use std version
			gt = self.gts_nmd[idx]
		else:
			pred = self.preds[idx]  # use std version
			gt = self.gts[idx]

		if self.if_aug:
			deg_rng = self.deg_rng
			scal_rng = self.scal_rng
			deg_x = random.uniform(-deg_rng[0], deg_rng[0])
			deg_y = random.uniform(-deg_rng[1], deg_rng[1])
			deg_z = random.uniform(-deg_rng[2], deg_rng[2])
			scal = random.uniform(scal_rng[0], scal_rng[1])

			# rtm = R.from_euler('xyz', angles=[deg_x, deg_y, deg_z])     # other rotation doesn't help only y?
			# rtm = R.from_euler('xyz', angles=[0, deg_y, 0])     # only rotate y
			rtm = R.from_euler('xyz', angles=[0, 0, 0])     # only scale
			# print('matrix is', rtm.as_matrix())
			pred = rtm.apply(pred*scal)      # 17 x 3  new coordinate
			gt = rtm.apply(gt*scal)   # to new scale and transformation

		pred_tch = torch.from_numpy(pred.flatten()).float()     # to one dim
		gt_tch = torch.from_numpy(gt.flatten()).float()     # to one dim

		return pred_tch, gt_tch

	def __len__(self):
		return self.N




