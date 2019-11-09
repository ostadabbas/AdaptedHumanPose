import torch
from torch import nn


class WhtL1Loss(torch.nn.Module):
	def __init__(self, whtScal=100, clipMod='clip11', baseWht=1):
		super(WhtL1Loss, self).__init__()
		self.whtScal = whtScal
		self.clipMod = clipMod
		self.baseWht = baseWht

	def forward(self, src, tar):
		if self.clipMod == 'clip11':
			wht = (tar + 1) / 2 * self.whtScal + self.baseWht
		elif self.clipMod == 'clip01':
			wht = tar * self.whtScal + self.baseWht
		else:  # no processing only adds wht in
			wht = tar + self.baseWht
		loss = (torch.abs(src - tar) * wht).mean()
		return loss


class GANloss_vec(torch.nn.Module):
	'''
	PatchGAN calculator, given indicator vec, calculate the result
	'''

	def __init__(self, gan_mode='lsgan'):
		super(GANloss_vec, self).__init__()
		# self.register_buffer('real_label', torch.tensor(target_real_label))
		# self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		# elif gan_mode in ['wgangp']:
		# 	self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def __call__(self, prediction, tar_vec):
		'''
		for adversarial training, you should set tar_vec accordingly. eg, for D loss, set fake to 1, for G set fake to 0
		:param prediction:
		:param tar_vec:
		:return:
		'''
		tar_temp = torch.ones_like(prediction)      # float
		tar_temp *= tar_vec.view(-1,1,1,1)    # expand to nbch * x * y , broad to change tar_temp

		if self.gan_mode in ['lsgan', 'vanilla']:
			loss = self.loss(prediction, tar_temp)
		# elif self.gan_mode == 'wgangp':   # target real, more one more real
		# 	if target_is_real:
		# 		loss = -prediction.mean()
		# 	else:
		# 		loss = prediction.mean()
		return loss
