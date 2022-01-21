'''
AHuP , SAA model
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
from .base_model import BaseModel
from .resnet import ResNetBackbone
from .networks import define_D, init_net, D_SA
from .nadam import Nadam
from .criterion import GANloss_vec, GANloss_SA, L_D_SA


class HeadNet(nn.Module):
	def __init__(self, opts):
		self.inplanes = 2048
		self.outplanes = 256

		super(HeadNet, self).__init__()

		self.deconv_layers = self._make_deconv_layer(3)
		self.final_layer = nn.Conv2d(
			in_channels=self.inplanes,
			out_channels=opts.ref_joints_num * opts.depth_dim,
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


class SAA(BaseModel):
	'''
	task generation network for 3D human.
	todo save/load scheduler
	'''

	def __init__(self, opts, if_train=True):
		super(SAA, self).__init__(opts)
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
		self.joint_num = opts.ref_joints_num
		self.if_train = if_train   # when not train, no optimizer, not looking for target
		self.if_z = 0.      # if supervise z, initially 0, at epoch to set
		self.loss_T = torch.tensor([-1.])        # initialize  in case no needed to indicate not training any longer.
		self.loss_G_GAN = torch.tensor([-1.])
		self.loss_D = torch.tensor([-1.])
		# self.stgs_D = opts.stgs_D
		self.mode_D = opts.mode_D           #  SA, C, or C1
		self.pivot = opts.pivot

		if 'res' in opts.net_BB:
			self.netG = init_net(ResNetBackbone(opts.net_BB), init_type=opts.init_type, gpu_ids=self.gpu_ids)
			self.inplanes = 2048  # input planes, 8 x 2048?
		else:
			self.netG = None  #
			self.inplanes = 256  #
		self.netT = init_net(HeadNet(opts), init_type=opts.init_type, gpu_ids=self.gpu_ids)

		self.models.append(self.netG)       # always put D later
		self.models.append(self.netT)

		if 'adam' == opts.optimizer:
			optimT = torch.optim.Adam
		elif 'nadam' == opts.optimizer:
			optimT = Nadam      # default 2e-3  0.9, 0.999, here adam is 1e-3

		if len(opts.trainset) > 1 and opts.lmd_D >0:    # more than 1 set to distinguish
			self.if_D = True
		else:
			self.if_D = False

		if if_train:

			if optimT == Nadam:
				betas = (0.9, 0.999)        # for nadam default
			else:
				betas = (0.5, 0.999)        # for original default
			self.optimizer_G = optimT(self.netG.parameters(), lr=opts.lr, betas=betas)
			self.optimizer_T = optimT(self.netT.parameters(), lr=opts.lr, betas=betas)
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_T)

			if self.if_D:  # updating by D option
				self.loss_names += ['G_GAN', 'D']
				# if not self.if_SA:
				if 'SA' in self.mode_D: # semantic aware        only SA needs jt ch_out
					ch_out = self.joint_num    #
				else:
					ch_out = 1
				self.criterionGAN = L_D_SA(opts.gan_mode, mode_L=self.mode_D).to(self.device)

				if '1' in opts.mode_D:      # C1 or SA1
					self.netD = define_D(self.inplanes, 64, 'convD_C1', n_layers_D=opts.n_layers_D, init_type=opts.init_type, gpu_ids=opts.gpu_ids, ch_out=ch_out, kn_D = opts.kn_D)  # 2 layers , use pixel for single point without expanding,
				else:
					self.netD = define_D(self.inplanes, 64, 'convD', n_layers_D=opts.n_layers_D, init_type=opts.init_type, gpu_ids=opts.gpu_ids, ch_out=ch_out, kn_D = opts.kn_D)  # 2 layers , use pixel for single point without expanding,
				self.models.append(self.netD)

				self.optimizer_D = optimT(self.netD.parameters(), lr=opts.lr, betas=betas)
				self.set_requires_grad(self.netD, False)        # prevent updating during  G
				self.optimizers.append(self.optimizer_D)
			# auto shedulers
			self.gen_auto_schedulers()

	def load_bb_pretrain(self):
		'''
		load the pretrain Backbone
		:return:
		'''
		netG = self.netG    # ref to it
		if isinstance(netG, torch.nn.DataParallel):
			netG = netG.module
		netG.init_weights()       # load public well pretrained one, dataParalel no load pretrain

	def fix_G(self):
		'''
		freeze the backbone G net parameters, all G not require
		:return:
		'''
		self.if_fixG = True
		self.if_D =False        # control current state
		self.set_requires_grad(self.netG, requires_grad=False)
		# get rid of the loss name to get rid of the requirement

		print('net G fixed')

	def forward(self):
		'''
		simply the G-T loop to get coords:HM
		:return:
		'''
		self.G_fts = self.netG(self.img)    # bch*x* y,  SA version will be  list of 4  bchxchxhxw
		self.HM = self.netT(self.G_fts)     # real size 256 512 1024 2048 ??!
		self.coord = soft_argmax(self.HM, self.joint_num)       # coord in HM

	def backward_D(self):
		'''
		forward D , D loss then back, optD step
		:return:
		'''
		# D_SA needs to calculate all the list posistions
		# netD(G_fts) input list of features, return list of features
		# li_rst_D  against the if_SYN,

		G_fts_de = self.G_fts.detach()
		pred_D = self.netD(G_fts_de)  # no affecting G expect 4d weight 128,64, 3,3, input 3d  [256, 64, 64]
		whts = self.whts_D  # set the weights D later, should be same size  N x hm
		if self.opts.mode_D == 'C1':    # other options can be extended here
			pred_pch, pred_C1 = pred_D[0], pred_D[1]
			arg_in_D = {'preds': pred_C1, 'if_SYNs': self.tgt_if_SYNs, 'whts': whts}       # 1 classification. and criterion
			self.loss_D = self.criterionGAN(arg_in_D)  # try to make syn one
		elif self.mode_D == 'SA1':     # combine SA and C1 together
			pred_pch, pred_C1 = pred_D[0], pred_D[1]
			arg_in_D = {'preds': pred_C1, 'if_SYNs': self.tgt_if_SYNs, 'whts': whts}  # 1 classification. and criterion
			loss_C1 = self.criterionGAN(arg_in_D)  # try to make syn one
			arg_in_D = {'preds': pred_pch, 'if_SYNs': self.tgt_if_SYNs, 'whts': whts}  # 1 classification. and criterion
			loss_pch = self.criterionGAN(arg_in_D)  # try to make syn one
			self.loss_D = loss_C1 + loss_pch        # together
			# combine the reuslt together
		else:       # all other cases, single pred_D
			arg_in_D = {'preds': pred_D, 'if_SYNs': self.tgt_if_SYNs, 'whts': whts}        # 16 ch wht, all in
			self.loss_D = self.criterionGAN(arg_in_D)  # try to make syn one

		self.loss_D.backward()

	def backward_G(self):
		'''
		from coord, get T_loss , forward D, get D loss.
		optimizer_G.zero_grad(), G_loss back, optT step, if not fix, optG,T step
		:return:
		'''
		self.loss_T_tot = torch.tensor([0.]).to(self.device) # zero loss
		if self.if_D:
			pred_D = self.netD(self.G_fts)  # avoid affecting G. Wrong should attatch
			whts = self.whts_D  # opt in loss funct

			if self.pivot == 'sdt':     # sdt specific syn rev
				if_SYNs_rev = torch.ones_like(self.tgt_if_SYNs)*0.5        # all even
			else:       # for both uda, and  inverse label
				if_SYNs_rev = 1 - self.tgt_if_SYNs

			if self.pivot == 'n':       # only n keep real, idx
				idx_smpl = self.tgt_if_SYNs.view(-1) == 0  # only keeps the real ones
			else:    # keep all
				N = self.tgt_if_SYNs.shape[0]
				idx_smpl = torch.tensor([True] * N, dtype=bool)     # similar to above
				# arg_in_D = {'preds': pred_D, 'if_SYNs': if_SYNs_rev, 'whts': whts}  # 16 ch wht    reverse if syn order wrong

			if self.mode_D == 'C1':     # C1 branche
				pred_pch, pred_C1 = pred_D[0], pred_D[1]
				arg_in_D = {'preds': pred_C1[idx_smpl], 'if_SYNs': if_SYNs_rev[idx_smpl], 'whts': whts[idx_smpl]}
				self.loss_G_GAN = self.criterionGAN(arg_in_D) * self.opts.lmd_D  # to make real one
			elif self.mode_D == 'SA1':
				pred_pch, pred_C1 = pred_D[0], pred_D[1]
				arg_in_D = {'preds': pred_C1[idx_smpl], 'if_SYNs': if_SYNs_rev[idx_smpl], 'whts': whts[idx_smpl]}
				loss_C1 = self.criterionGAN(arg_in_D) * self.opts.lmd_D
				arg_in_D = {'preds': pred_pch[idx_smpl], 'if_SYNs': if_SYNs_rev[idx_smpl], 'whts': whts[idx_smpl]}
				loss_pch = self.criterionGAN(arg_in_D) * self.opts.lmd_D
				self.loss_G_GAN = loss_C1 + loss_pch
			else:
				arg_in_D = {'preds': pred_D[idx_smpl], 'if_SYNs': if_SYNs_rev[idx_smpl], 'whts': whts[idx_smpl]}
				self.loss_G_GAN = self.criterionGAN(arg_in_D) * self.opts.lmd_D # to make real one

			self.loss_T_tot += self.loss_G_GAN

		# task loss, back
		loss_coord = torch.abs(self.coord - self.tgt_coord) * self.tgt_vis  # vis based loss, original 3DMPPE gives abs

		loss_coord = (loss_coord[:,:,0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * self.tgt_have_depth * self.if_z) /3.      # get rid of z part
		self.loss_T = loss_coord.mean()
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
		if self.if_train and target:    # train model there is target , then set
			self.tgt_coord = target['joint_hm'].to(self.device)
			self.tgt_vis = target['vis'].to(self.device)
			self.tgt_have_depth = target['if_depth_v'].to(self.device)
			self.tgt_if_SYNs = target['if_SYN_v'].to(self.device)       # 120 x 1
			self.whts_D = target['wts_D'].to(self.device)  # give weights D in feeder

	def optimize_parameters(self):
		self.forward()
		if self.if_D:
			self.set_requires_grad(self.netD, True)  # enable backprop for D
			self.optimizer_D.zero_grad()  # set D's gradients to zero
			self.backward_D()  # calculate gradients for D
			self.optimizer_D.step()  # update D's weights
			self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
		self.optimizer_G.zero_grad()  # set G's gradients to zero
		self.optimizer_T.zero_grad()
		self.backward_G()  # T, G both, G include GAN(D) loss, will affect D
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
