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

	def __init__(self, gan_mode='lsgan', reduction='mean'):
		super(GANloss_vec, self).__init__()
		# self.register_buffer('real_label', torch.tensor(target_real_label))       # persiste buffer, a state but no parameter, but grad I think
		# self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss(reduction=reduction)
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
		# elif gan_mode in ['wgangp']:
		# 	self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def __call__(self, arg_in_D):
		'''
		for adversarial training, you should set tar_vec accordingly. eg, for D loss, set fake to 1, for G set fake to 0
		:param prediction:
		:param tar_vec:
		:return:
		'''
		prediction = arg_in_D['preds']
		tar_vec = arg_in_D['if_SYNs']

		tar_temp = torch.ones_like(prediction)      # float
		tar_temp *= tar_vec.view(-1,1,1,1)    # expand to nbch * x * y , broad to change tar_temp

		if self.gan_mode in ['lsgan', 'vanilla']:
			loss = self.loss(prediction, tar_temp)
		else:
			raise ValueError('{} mode not implemented'.format(self.gan_mode))
		# elif self.gan_mode == 'wgangp':   # target real, more one more real
		# 	if target_is_real:
		# 		loss = -prediction.mean()
		# 	else:
		# 		loss = prediction.mean()
		return loss


class L_D_SA(torch.nn.Module):
	'''
	loss D , SA, compatible with conventional
	'''

	def __init__(self, gan_mode='lsgan', mode_L='C'):
		super(L_D_SA, self).__init__()
		# self.register_buffer('real_label', torch.tensor(target_real_label))
		# self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		self.mode_L = mode_L        # the mode of the operation
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss(reduction='none')
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss(reduction='none')
		# elif gan_mode in ['wgangp']:
		# 	self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def __call__(self, arg_in_D):
		'''
		for adversarial training, you should set tar_vec accordingly. eg, for D loss, set fake to 1, for G set fake to 0
		:param prediction:
		:param tar_vec:
		:return:
		'''
		prediction = arg_in_D['preds']
		tar_vec = arg_in_D['if_SYNs']
		wht = arg_in_D['whts']
		# preds_C1 = arg_in_D['preds_C1']

		# loss_tot = torch.Variable
		# if prediction is not None:
		tar_temp = torch.ones_like(prediction)  # float
		shp_tar = [-1] + [1] * (prediction.ndimension()-1)
		tar_temp *= tar_vec.view(*shp_tar)  # expand to nbch * x * y , broad to change tar_temp
		ch=prediction.shape[1]      # channel number
		# print('pred shape', prediction.shape)       # dbg , 120 and 80 for all or real
		if self.gan_mode in ['lsgan', 'vanilla']:
			loss = self.loss(prediction, tar_temp)
		else:
			raise ValueError('{} mode not implemented'.format(self.gan_mode))
		if self.mode_L =='SA':
			# print("SA criterion triggered")
			loss = loss * wht
			return loss.sum()/ch    # average over channel, similar weight as C version
		return loss.mean()  # average all

class GANloss_SA(torch.nn.Module):      # obsolete for guassian SA
	'''
	for D_SA criterion, multi-stage gaussian
	'''
	def __init__(self, gan_mode='lsgan'):
		super(GANloss_SA, self).__init__()
		if gan_mode not in ['lsgan', 'vanilla']:
			raise ValueError('{} mode not implemented'.format(self.gan_mode))
		self.D_vec=GANloss_vec(gan_mode=gan_mode, reduction='none') # keep original

	def __call__(self, args):
		'''
		from D_SA result, everyone compare with if_SYN, loop valid stg,  if li_gs, then filter the rst, and add them together. args is a dict including:
		:param li2_D:
		:param if_SYN:
		:param stgs_D:
		:param li2_gs: gaussian hm filter, if not SA, use none to give full map discrimination
		:return:
		'''
		# check tensor device to create the
		li2_D = args['li2_D']
		if_SYNs = args['if_SYNs']
		stgs_D = args['stgs_D']
		li2_gs = args['li2_gs']
		li_rst = []
		# n_jt = len(li2_D[0]) # how many jts
		for i, if_D_stg in enumerate(stgs_D):
			n_jt = len(li2_D[i])
			if if_D_stg:
				for j in range(n_jt):
					rst_t = self.D_vec(li2_D[i][j], if_SYNs)    # no collade  yet i stage, j joint
					if li2_gs:   # if there is hm
						# print('li2_gs type', type(li2_gs[i][j]))
						# print('rst_t type', type(rst_t))
						rst_t = rst_t * li2_gs[i][j].float()    # rst i singular yet gs is not, expect doulbe but get float?
					li_rst.append(torch.sum(rst_t))
		return sum(li_rst) /len(li_rst)    # default ave, but will balance out large area, use average, n_stg x n_jt

# make GAN
