'''
model and operation design for task generation
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.resnet import ResNetBackbone

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from .base_model import BaseModel
from .resnet import ResNetBackbone
from .networks import define_D, init_net
from .nadam import Nadam
from .criterion import GANloss_vec

import opt


class HeadNet(nn.Module):
	def __init__(self, opts):
		self.inplanes = 2048
		self.outplanes = 256

		super(HeadNet, self).__init__()

		self.deconv_layers = self._make_deconv_layer(3)
		self.final_layer = nn.Conv2d(
			in_channels=self.inplanes,
			out_channels=opts.joint_num * opts.depth_dim,
			kernel_size=1,
			stride=1,
			padding=0
		)

	def _make_deconv_layer(self, num_layers):
		layers = []
		for i in range(num_layers):
			layers.append(
				nn.ConvTranspose2d(
					in_channels=self.inplanes,
					out_channels=self.outplanes,
					kernel_size=4,
					stride=2,
					padding=1,
					output_padding=0,
					bias=False))
			layers.append(nn.BatchNorm2d(self.outplanes))
			layers.append(nn.ReLU(inplace=True))
			self.inplanes = self.outplanes

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.deconv_layers(x)
		x = self.final_layer(x)

		return x

	def init_weights(self):
		for name, m in self.deconv_layers.named_modules():
			if isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight, std=0.001)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		for m in self.final_layer.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, std=0.001)
				nn.init.constant_(m.bias, 0)


class TaskGenNet(BaseModel):
	'''
	task generation network for 3D human.
	todo save/load scheduler
	'''

	def __init__(self, opts, if_train=True):
		super(TaskGenNet, self).__init__(opts)
		# backbone(G), headnet(T, task), netD
		# optimizer G, T, D
		# if continue, find latest epoch_start, otherwise 0
		# define the loss here T_loss, D_loss
		# start_epoch set self.if_fixG, at beginning each epoch check and set it.
		# self.opts = opts  # keep system opt, will be udpated as ref outside
		# self.if_fixG = True if 'y' == opts.if_fixG else False
		self.if_fixG = False    # current state
		self.model_names = ['G', 'T']
		self.loss_names = ['T']
		self.visuals = ['G_fts']  # only show G features
		self.joint_num = opts.joint_num
		self.if_train = if_train   # when not train, no optimizer, not looking for target
		self.if_z = 0.      # if supervise z, initially 0, at epoch to set

		if 'res' in opts.net_BB:
			self.netG = init_net(ResNetBackbone(opts.net_BB), init_type=opts.init_type, gpu_ids=self.gpu_ids)
			self.inplanes = 2048  # input planes
		else:
			self.netG = None  # todo --- fill if other net needs
			self.inplanes = 256  # todo --
		self.netT = init_net(HeadNet(opts), init_type=opts.init_type, gpu_ids=self.gpu_ids)

		self.models.append(self.netG)       # always put D later
		self.models.append(self.netT)

		if 'adam' == opts.optimizer:
			optimT = torch.optim.Adam
		elif 'nadam' == opts.optimizer:
			optimT = Nadam


		if len(opts.trainset) > 1 and opts.lmd_D >0 :
			self.if_D = True


		if if_train:
			# criterion, T use abs directly
			# optimizer
			self.optimizer_G = optimT(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
			self.optimizer_T = optimT(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_T)

			if self.if_D:  # updating by D option
				self.loss_names += ['G_GAN', 'D']
				self.netD = define_D(self.inplanes, 64, 'n_layers', n_layers=opts.n_layers_D, init_type=opts.init_type, gpu_ids=opt.gpu_ids)
				self.models.append(self.netD)
				# self.criterionGAN = GANLoss(opts.gan_mode).to(self.device)
				self.criterionGAN = GANloss_vec(opts.gan_mode).to(self.device)
				self.optimizer_D = optimT(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
				self.optimizers.append(self.optimizer_D)
			# auto shedulers
			self.gen_auto_schedulers()

	def load_bb_pretrain(self):
		'''
		load the pretrain Bb
		:return:
		'''
		self.netG.loadPreTrain(str(self.device))       # load public well pretrained one

	def fix_G(self):
		'''
		freeze the backbone G net parameters, all G not require
		:return:
		'''
		self.if_fixG = True
		self.if_D =False
		self.set_requires_grad(self.netG, requires_grad=False)

	def forward(self):
		'''
		simply the G-T loop to get coords:HM
		:return:
		'''
		self.G_fts = self.netG(self.img)    # bch*x* y
		self.HM = self.netT(self.G_fts)     # 64 * 64  * (17* 64)
		self.coord = soft_argmax(self.HM, self.joint_num)       # coord in HM

	def backward_D(self):
		'''
		forward D , D loss then back, optD step
		:return:
		'''
		pred_D = self.netD(self.G_fts.detch())  # all prediction rst
		self.loss_D = self.criterionGAN(pred_D, self.tgt_if_SYNs) * self.opts.lmd_D
		self.loss_D.backward()

	def backward_G(self):
		'''
		from coord, get T_loss , forward D, get D loss.
		optimizer_G.zero_grad(), G_loss back, optT step, if not fix, optG,T step
		:return:
		'''
		self.loss_T_tot = torch.tensor([0.]) # zero loss
		if self.if_D:
			pred_D = self.netD(self.G_fts.detach())  # avoid affecting G
			self.loss_G_GAN = self.criterionGAN(pred_D, 1 - self.tgt_if_SYNs) * self.opts.lmd_D
			self.loss_T_tot += self.loss_G_GAN

		# task loss, back
		loss_coord = torch.abs(self.coord - self.tgt_coord) * self.tgt_vis  # vis based loss
		loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * self.tgt_have_depth * self.if_z) /3.      # get rid of z part
		self.loss_T = loss_coord
		self.loss_T_tot += self.loss_T
		self.loss_T_tot.backward()

	# update_learning_rate() in base

	def set_input(self, input, target=None):
		'''
		:param input:
		:param target: is the supervision signal
		:return:
		'''
		self.img = input['img_patch']   # when used for wild image only input
		if self.if_train:
			if not target:
				print('error, no target set as input')
			else:
				self.tgt_coord = target['joint_hm']
				self.tgt_vis = target['vis']
				self.tgt_have_depth = target['if_depth_v']
				self.tgt_if_SYNs = target['if_SYN_v']

	def optimize_parameters(self):
		self.forward()
		if 'y' == self.if_D:
			self.set_requires_grad(self.netD, True)  # enable backprop for D
			self.optimizer_D.zero_grad()  # set D's gradients to zero
			self.backward_D()  # calculate gradients for D
			self.optimizer_D.step()  # update D's weights
		# update G
		self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
		self.optimizer_G.zero_grad()  # set G's gradients to zero
		self.optimizer_T.zero_grad()
		self.backward_G()  # T, G both
		if not self.if_fixG:
			self.optimizer_G.step()
		self.optimizer_T.step()

def soft_argmax(heatmaps, joint_num, depth_dim=64, output_shape=(64, 64)):
	heatmaps = heatmaps.reshape((-1, joint_num, depth_dim * output_shape[0] * output_shape[1]))
	heatmaps = F.softmax(heatmaps, 2)
	heatmaps = heatmaps.reshape((-1, joint_num, depth_dim, output_shape[0], output_shape[1]))

	accu_x = heatmaps.sum(dim=(2, 3))
	accu_y = heatmaps.sum(dim=(2, 4))
	accu_z = heatmaps.sum(dim=(3, 4))

	accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, output_shape[1] + 1).type(torch.cuda.FloatTensor),
	                                            devices=[accu_x.device.index])[0]
	accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, output_shape[0] + 1).type(torch.cuda.FloatTensor),
	                                            devices=[accu_y.device.index])[0]
	accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1, depth_dim + 1).type(torch.cuda.FloatTensor),
	                                            devices=[accu_z.device.index])[0]

	accu_x = accu_x.sum(dim=2, keepdim=True) - 1
	accu_y = accu_y.sum(dim=2, keepdim=True) - 1
	accu_z = accu_z.sum(dim=2, keepdim=True) - 1

	coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

	return coord_out
