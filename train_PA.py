'''
train the PA net, for pred -> gt direct mapping, no adaptation.
'''

import torch
from opt import opts, set_env
from models.smplBL import  PAnet, weight_init
from torch import nn
from utils.logger import Colorlogger
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Qt4Agg')   # or just Agg  for non-interactive
import os
import os.path as osp
from data.P3D import P3D
from torch.utils.data import DataLoader
from time import time
import numpy as np
import utils.utils_pose  as ut_p
from tqdm import tqdm
import json

# model
# log
# if resume load model
# ds init according to opt, make loader, pred, tar read in file
# test loader, from test session,  load the test files. h36, muco-> MuCoTS, geo, if train add transformaer
# optimizer, loss
# for epoch ,
# print the MPJPE , loss
# clean up the name test

def loop(model, DL, epoch, phase='train', logger=None, **kwargs):
	'''
	ran the loop , according to the phase do train or test .
	:param model:
	:param DL:
	:param logger:
	:param phase:
	:param kwargs: keep the optmizer
	:return:
	'''
	if logger:
		f_prt = logger.info
	else:
		f_prt = print
	# keep epoch time
	ds = DL.dataset
	if phase == 'train':
		optim = kwargs['optim']
		criterion = kwargs['criterion']
		# scheduler = kwargs['scheduler']   # epoch level
		n_iter = opts.trainIter
		model.train()
	else:
		n_iter = opts.testIter
		criterion = kwargs['criterion']
		model.eval()

	epoch_st_tm = time()
	st_tm = time()
	li_loss = []
	li_out = []
	for i, (inputs, tars )in enumerate(DL):
		cost_tm = time() - st_tm
		st_tm = time()
		inputs = inputs.cuda()
		tars = tars.cuda()

		if n_iter>0 and i>= n_iter:
			break
		outputs = model(inputs)
		li_out.append(outputs.detach().cpu().numpy())

		if phase == 'train':
			# missing 0 gradient
			optim.zero_grad()
			loss = criterion(outputs, tars)
			loss.backward()
			nn.utils.clip_grad_norm(model.parameters(), max_norm=1) # clip the norm
			optim.step()        #

		else:
			loss = criterion(outputs, tars)     # same thing for test here

		# updaing the msg
		loss_flt = float(loss)      # should be almost to MPJPE
		li_loss.append(loss_flt)
		if i % opts.print_freq == 0:
			msg_li = [
				'Epoch {:3} iter {:5}/{:5}'.format(epoch, i, len(DL)),
				'time {:5.3f}ms/{}'.format(cost_tm*1000, opts.print_freq),
				'loss {:5.3f}'.format(loss_flt)
			]
			f_prt(' '.join(msg_li))
	gts = ds.gts        # ori   N x 17 x 3
	preds = np.concatenate(li_out, axis=0).reshape(gts.shape)
	if opts.if_norm_PA == 'y':      # if not normed no need to take care
		preds = preds*ds.std_pred + ds.mu_pred     # the ori
	# pred_3d_kpt_align = ut_p.rigid_align(pred_3d_kpt, gt_3d_kpt) ge align
	preds_align = []
	if phase == 'test':
		for pred, gt in tqdm(zip(preds, gts), desc='>>>aligning preds '):
			pred_align = ut_p.rigid_align(pred, gt)
			preds_align.append(pred_align)
		preds_align = np.array(preds_align)  #
		diff_align = preds_align - gts
		av_align_err = np.mean(np.power(np.sum(diff_align ** 2, axis=2), 0.5))  # MPJPE
	else:
		av_align_err = -1       # for position holder not calculated

	diff = preds - gts
	av_err = np.mean(np.power(np.sum(diff**2, axis=2), 0.5))        # MPJPE
	av_loss = np.array(li_loss).mean()

	# give alignments
	f_prt('{:5} epoch {:3} time:{:5.3f} av_loss: {:5.3f}, av_err:{:5.3f} av_align_err:{:4.3f}'.format(phase, epoch, time()-epoch_st_tm, av_loss, av_err, av_align_err))

	return av_loss, av_err, preds


def main():
	set_env(opts)       # set the  path etc
	tarset = opts.tarset_PA
	bch_sz = opts.batch_size_PA         # 256

	model = PAnet(d=3)      # 3d input 51
	model = model.cuda()
	model.apply(weight_init)
	sv_hd = osp.join(opts.rst_dir, 'ckpt_{}_VPR-{}'.format(tarset, opts.if_VPR))  # ckpt_h36m-p1.pth

	criterion = nn.MSELoss().cuda()     # size_average=True deprecated
	optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opts.decay_PA),ori  0.96 gamma 100000 steps
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.85)    # every epoch , step down
	err_best = 1000     # to keep the best err here
	logger = Colorlogger(opts.log_dir, 'PA_{}_logs.txt'.format(opts.tarset_PA))
	start_epoch = 0
	logger.info('>>> working dict to {}'.format(opts.exp_dir))
	if opts.if_continue_PA == 'y':
		logger.info(">>> try to load previous checkpoints")
		ckpt_pth = osp.join(opts.model_dir,  'ckpt_PA_{}.pth'.format(tarset))   # saved for model specific
		if osp.exists(ckpt_pth):
			ckpt = torch.load(ckpt_pth)
			start_epoch = ckpt['epoch']      # from next epoch
			err_best = ckpt['err']
			# glob_step = ckpt['step']
			# lr_now = ckpt['lr']
			model.load_state_dict(ckpt['state_dict'])
			optimizer.load_state_dict(ckpt['optimizer'])
			scheduler.load_state_dict(ckpt['scheduler'])
			logger.info(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
		else:
			logger.info(">>> no ckpt exists")

	ds_train = P3D(opts, split='train')     # get the proper ds
	ds_test = P3D(opts, split='test')
	train_loader = DataLoader(
		dataset=ds_train,
		batch_size=bch_sz,
		shuffle=True,
		num_workers=opts.n_thread,
		pin_memory=True)
	test_loader = DataLoader(
		dataset=ds_test,
		batch_size=bch_sz,
		shuffle=False,
		num_workers=opts.n_thread,
		pin_memory=True)
	epoch = start_epoch
	if opts.if_test_PA == 'n':	        # if this run for test no
		loss_li = []
		err_li = []
		for epoch in range(start_epoch, opts.end_epoch_PA):
			# loop train
			loss, err, _ = loop(model, train_loader, epoch, phase='train', logger=logger, criterion=criterion, optim=optimizer)
			loss_li.append(loss)
			loss_test, err_test, _ = loop(model, test_loader, epoch, phase='test', logger=logger, criterion=criterion, optim=optimizer)
			scheduler.step()
			# can add saving point for freq, but  save the trouble it is quick
			is_best = err_test < err_best       # the mpjpe
			sv_dict = {
				'epoch': epoch + 1,  # to next epoch when start again
				'err': err_test,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()
			}
			sv_pth = sv_hd +'.pth'
			logger.info('>>> save model to {}'.format(sv_pth))
			torch.save(sv_dict, sv_pth)
			if is_best:     # best save again
				err_best = err_test # to current one
				sv_pth = sv_hd + '_best.pth'
				sv_dict = {
					'epoch': epoch + 1,  # to next epoch when start again
					'err': err_best,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict()
				}
				logger.info('>>> save best model to {}'.format(sv_pth))
				torch.save(sv_dict, sv_pth)
		loss_arr = np.array(loss_li)
		plt.plot(loss_arr)
		plt.show()
		# save final model, no need

	logger.info("err best is {}".format(err_best))
	# loop final test report
	loss_test, err_test, preds = loop(model, test_loader, epoch, phase='test', logger=logger, criterion=criterion, optim=optimizer)
	if opts.if_hm_PA == 'y': # if adapt at pm level  save output ad preds_hm results

		# hardwire pth_hd
		if tarset == 'h36m-p1':
			pth_hd = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto1'
		elif tarset == 'h36m-p2':
			pth_hd = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto2'
		elif tarset == 'MuCo':
			pth_hd = 'MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test'        # MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test_pred_hm.npy
		pth_hd = osp.join(opts.rst_dir, pth_hd)
		# if 'h36m' in opts.tarset_PA:  # split_flip-[y|n]_e{epoch}_[proto1/2]
		# 	proto = int(float(opts.tarset_PA[-1]))      #
		# 	pth_hd = osp.join(opts.rst_dir,
		# 	                  '_'.join([opts.nmTest, ds_test.data_split, 'proto' + str(proto)])) #
		# else:
		# 	pth_hd = osp.join(opts.rst_dir, '_'.join([opts.nmTest, ds_test.data_split]))
		sv_pth = pth_hd +'_pred_hm_PA.npy'     # Human36M_Btype-h36m_SBL-n_PA-n_exp_train_proto2_pred_camRt.json
		logger.info('saving hte PA test pm to {}'.format(sv_pth))
		np.save(sv_pth, preds)      #



if __name__ == '__main__':
	main()