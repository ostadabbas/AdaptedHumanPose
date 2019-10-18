'''
model and operation design for task generation
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone

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

import opt

class TaskGen(BaseModel):
	'''
	task generation network for 3D human.
	'''
	def __init__(self, opt, if_train=True):
		super(TaskGen, self).__init__()
		# backbone(G), headnet(T, task), netD
		# optimizer G, T, D
		# if continue, find latest epoch_start, otherwise 0
		# define the loss here T_loss, D_loss
		# start_epoch set self.if_fixG, at beginning each epoch check and set it.
		self.opt = opt  # keep system opt, will be udpated as ref outside
		self.if_fixG = False    # change later



	def freeze_G(self):
		'''
		freeze the backbone G net parameters, all G not require
		:return:
		'''
		self.if_fixG = True
		# self.set_requires_grad(netG)

	def forward(self):
		'''
		simply the G-T loop to get coords:HM
		:return:
		'''
		pass

	def backward_G(self):
		'''
		from coord, get T_loss , forward D, get D loss.
		optimizer_G.zero_grad(), G_loss back, optT step, if not fix, optG,T step
		:return:
		'''
		pass

	def backward_D(self):
		'''
		forward D , D loss then back, optD step
		:return:
		'''
		pass

	def update_lr(self):
		'''
		according to current epoch, update lr rate of all optimizers
		:return:
		'''
		pass

	def setup_input(self, input):
		self.img = input

	def optimize_parameters(self):
		self.forward()
		if 'y' == self.opt.if_D:
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