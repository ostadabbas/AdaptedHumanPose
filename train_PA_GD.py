'''
train the PA GD net, specify the src(gt or pred) tar,  all gt right now.
All possible compoennts are provided. We only use SPA strategy at the moment.
'''

import torch
from opt import opts, set_env

from models.smplBL import  PAnet, weight_init, PA_D
from torch import nn
from utils.logger import Colorlogger
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Qt4Agg')   # or just Agg  for non-interactive
import os
import os.path as osp
from data.P3D import P3D, P3D_D
from torch.utils.data import DataLoader
from time import time
import numpy as np
import utils.utils_pose  as ut_p
import utils.utils_tool  as ut_t
from tqdm import tqdm
import json
import argparse
from utils.evaluate import get_MPJPE

## hardwired env variables, can be in parser,
dt_fd = 'data_files'        # keep the gt data
lmd_D = 1.
lmd_G = 1e-3        # hm about 64 so largest 30 * 30 = 900, can be tune
idx_pelvis = 0
idx_thorax = 8

# save name
exp_dir = 'output/GD_PA'  # GD_PA folder independent
model_dir = osp.join(exp_dir, 'model_dump')
logger_dir = osp.join(exp_dir, 'log')
rst_dir = osp.join(exp_dir, 'result')       # save to GD_PA for  shared
ut_t.make_folder(logger_dir)
ut_t.make_folder(model_dir)
ut_t.make_folder(rst_dir)
logger = Colorlogger(logger_dir, 'GD_logs.txt') # global logger

def get_hm_arr_PA(opts):
	'''
	from the current setting and tarset_PA, get the hm_arr for source and tar
	:param opts:
	:return:
	'''
	tarset_PA = opts.tarset_PA
	if tarset_PA == 'h36m-p1':
		dsNm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto1_pred_hm'
	elif tarset_PA == 'h36m-p2':
		dsNm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto2_pred_hm'
	elif tarset_PA == 'MuPoTS' or tarset_PA == 'MuCo':      # same test data
		dsNm = 'MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test_pred_hm'

	dt_pth = osp.join(opts.rst_dir, dsNm+'.json')
	logger.info('>>> loading test data from {}'.format(dt_pth))
	with open(dt_pth, 'r') as f:
		dtIn = json.load(f)
		f.close()
	preds = np.array(dtIn['pred'])  # pred gt of the h36m trainset
	gts = np.array(dtIn['gt'])

	return preds, gts


def loop(models, DL, epoch, phase='train', logger=None, **kwargs):
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
	G = models[0]
	D = models[1]
	ds = DL.dataset
	if opts.if_skel_MSE == 'y':
		critSkel = nn.MSELoss().cuda()
	else:
		critSkel = nn.L1Loss().cuda()

	if phase == 'train':
		optims = kwargs['optims']
		optimG = optims[0]
		optimD = optims[1]
		crits = kwargs['crits']
		critG = crits[0]
		critD = crits[1]
		# scheduler = kwargs['scheduler']   # epoch level
		n_iter = opts.trainIter
		G.train()
		D.train()
	else:
		n_iter = opts.testIter
		crits = kwargs['crits']
		critG = crits[0]
		critD = crits[1]
		G.eval()
		D.eval()

	epoch_st_tm = time()
	st_tm = time()
	li_loss_D = []        # for D loss here
	li_loss_L = []        # for D loss here
	li_loss_skel = []        # for D loss here
	li_out = []     # for pred later
	# for i, (inputs, tars)in enumerate(DL):
	for i, inps in enumerate(DL):
		inputs = inps[0]
		tars = inps[1]
		skels_inp = inps[2]
		skels_tar = inps[3]

		cost_tm = time() - st_tm
		st_tm = time()
		inputs = inputs.cuda()
		tars = tars.cuda()
		skels_tar = skels_tar.cuda()

		if False:
			print('inputs require grad', inputs.requires_grad)      # all false grad
			print('tars require grad', tars.requires_grad)
		if n_iter>0 and i>= n_iter:
			break
		preds = G(inputs)
		# get the skel
		preds_jt = preds.view(-1, 17,3)     # fixed size
		skels_pred= ut_p.get_l_skels_tch(preds_jt, ds.skeleton, ds.skel_norm_pair)

		li_out.append(preds.detach().cpu().numpy())
		out_src = D(preds)  # mis match 10 x 50  100 x 1
		out_tar = D(tars)  # N x1
		src_label = torch.tensor(1.0, requires_grad=True).expand_as(
			out_src).cuda()  # or syn label, must be sequence get float??
		tar_label = torch.tensor(0.0, requires_grad=True).expand_as(out_tar).cuda()

		if phase == 'train':
			# D backward, can free and freeze grad, but for G if only optimG step, no need to care, only for efficiency I think  .
			optimD.zero_grad()
			loss_D_src = critD(out_src.detach(), src_label)        #
			loss_D_tar = critD(out_tar.detach(), tar_label)        #
			loss_D = loss_D_src + loss_D_tar
			loss_D.backward()       #   # no grad func , maybe the label requrie_grad=True
			optimD.step()

			# G updating.
			optimG.zero_grad()
			optimD.zero_grad()
			loss_GAN = critD(out_src, tar_label) # to real one?
			loss_L = critG(preds, inputs)  # not too far away from original
			loss_skel = critSkel(skels_pred, skels_tar)  #
			loss_G = loss_GAN*opts.lmd_D_PA + loss_L*opts.lmd_G_PA + opts.lmd_skel_PA * loss_skel
			loss_G.backward()
			if opts.if_clip_grad == 'y':
				nn.utils.clip_grad_norm(G.parameters(), max_norm=1)  # clip the norm, for G right now
			optimG.step()

		else:       # test only get loss

			loss_D_src = critD(out_src, src_label)  #   not employed !
			loss_D_tar = critD(out_src, tar_label)  #
			loss_D = loss_D_src + loss_D_tar        # no back propo
			loss_L = critG(preds, inputs)   # preds and inputs!!?
			loss_skel = critSkel(skels_pred, skels_tar)  # not too far away from original

		# updaing the msg
		loss_flt_D= float(loss_D)      # simialr mpjpe in hm
		loss_flt_L = float(loss_L)      # simialr mpjpe in hm # check the l1 loss
		loss_flt_skel = float(loss_skel)      # simialr mpjpe in hm # check the l1 loss
		li_loss_L.append(loss_flt_L)
		li_loss_D.append(loss_flt_D)
		li_loss_skel.append(loss_flt_skel)

		if i % opts.print_freq == 0:
			msg_li = [
				'Epoch {:3} iter {:5}/{:5}'.format(epoch, i, len(DL)),
				'time {:5.3f}ms/{}'.format(cost_tm*1000, opts.print_freq),
				'loss_L {:5.3f}'.format(loss_flt_L),
				'loss_skel {:5.3f}'.format(loss_flt_skel),
				'loss_D {:5.3f}'.format(loss_flt_D),
			]
			f_prt(' '.join(msg_li))

	if phase == 'test':
		gts = ds.gts        # ori   N x 17 x 3, neck centered
		inputs = ds.preds       # weak labels
		preds = np.concatenate(li_out, axis=0).reshape(gts.shape)   # should be same shape? why miss so muhc
		if opts.if_neckRt == 'y':
			preds = preds + ds.preds_thrx       # recover back
		av_err, av_align_err = get_MPJPE(preds, gts)    # mapped -> gt
		err_ori, err_align_ori = get_MPJPE(inputs, gts)     # weak -> gt
		av_err_skel_ori = np.abs(ds.l_skel_inp - ds.l_skel_tar).mean()
	else:
		av_err = -1.    # not available
		av_align_err = -1.
		err_ori = -1
		err_align_ori = -1


	av_loss_G = np.array(li_loss_L).mean()
	av_loss_skel = np.array(li_loss_skel).mean()
	av_loss_D = np.array(li_loss_D).mean()
	# give alignments
	if phase == 'test':
		f_prt('{:5} epoch {:3} time:{:5.3f}, av_loss_G: {:5.3f}, av_loss_skel: {:5.3f},av_loss_D: {:5.3f}, av_err:{:5.3f} av_align_err:{:4.2f} err_ori:{:5.3f} err_aign_ori:{:5.3f} err_skel_ori:{:5.3f} '.format(phase, epoch, time()-epoch_st_tm, av_loss_G, av_loss_skel, av_loss_D,  av_err, av_align_err, err_ori, err_align_ori, av_err_skel_ori))
	else:
		f_prt('{:5} epoch {:3} time:{:5.3f} av_loss_G: {:5.3f} av_loss_skel: {:5.3f} av_loss_D: {:5.3f}'.format(phase, epoch,time() - epoch_st_tm, av_loss_G, av_loss_skel, av_loss_D))

	return [av_loss_G, av_loss_D], av_err, preds


def main():
	## hard wired env

	set_env(opts)
	bch_sz = opts.batch_size_PA

	# keep G, D loss
	G = PAnet(d=3, mode=opts.PA_G_mode)      # 3d input 51
	G = G.cuda()        # make G and D
	G.apply(weight_init)

	D = PA_D()
	D = D.cuda()
	D.apply(weight_init)

	tarset_PA = opts.tarset_PA
	dct_tarNm ={
		'h36m-p1': 'Human36M-p1',
		'h36m-p2': 'Human36M-p2',
		'MuCo': 'MuCo',
		'MuPoTS': 'MuPoTS'
	}
	srcNm = opts.trainset[0]      # ScanAva
	tarNm  = dct_tarNm[opts.tarset_PA]  # read from the ds files
	sv_hd = osp.join('{}_{}'.format(srcNm, tarNm))  #ScanAva_Human36-p1
	if opts.if_neckRt == 'y':
		sv_hd = sv_hd + '_neck'     # save with neck name

	# criterionG = nn.MSELoss().cuda()     # size_average=True deprecated
	if opts.if_L_MSE == 'y':
		criterionG = nn.MSELoss().cuda()
	else:
		criterionG = nn.L1Loss().cuda()     # optional

	criterionD = nn.BCEWithLogitsLoss().cuda()     # size_average=True deprecated

	optimizerG = torch.optim.Adam(G.parameters(), lr=opts.lr_PA)
	optimizerD  = torch.optim.Adam(D.parameters(), lr=opts.lr_PA)

	schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=opts.gamma_PA)    # every epoch , step down
	schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=opts.gamma_PA)    # every epoch , step down
	err_best = 1000
	start_epoch = 0

	ckpt_pth = osp.join(model_dir, sv_hd + '.pth')
	if opts.if_continue_PA == 'y' or opts.if_test_PA == 'y':  # load correconding state dict, load into two models.
		logger.info(">>> try to load previous checkpoints")
		if osp.exists(ckpt_pth):
			ckpt = torch.load(ckpt_pth)
			start_epoch = ckpt['epoch']      # from next epoch
			err_best = ckpt['err']      # keep the test error
			# glob_step = ckpt['step']
			# lr_now = ckpt['lr']
			G.load_state_dict(ckpt['G'])
			optimizerG.load_state_dict(ckpt['optim_G'])
			schedulerG.load_state_dict(ckpt['sch_G'])

			D.load_state_dict(ckpt['D'])
			optimizerD.load_state_dict(ckpt['optim_D'])
			schedulerD.load_state_dict(ckpt['sch_D'])
			logger.info(">>> ckpt loaded from {} epoch {}".format(ckpt_pth, start_epoch))
		else:
			logger.info(">>> no ckpt exists")
	src_pth = osp.join(dt_fd, '{}_hm.json'.format(srcNm))
	tar_pth = osp.join(dt_fd, '{}_hm.json'.format(tarNm))
	if opts.if_gtSrc == 'y':
		with open(src_pth, 'r') as f:
			dt_in = json.load(f)
			f.close()
		src_arr = np.array(dt_in)
	else:
		src_arr = np.load(osp.join(opts.rst_dir, '{}_Btype-h36m_SBL-n_PA-n_exp_train_pred_hm.npy'.format(srcNm)))

	with open(tar_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	tar_arr = np.array(dt_in)   #  list

	preds, gts = get_hm_arr_PA(opts)  # the pairded one
	ds_test = P3D_D(preds, gts, opts, split='test')  #

	if opts.if_gt_PA == 'y':        # use gt tar
		ds_train = P3D_D(src_arr, gts, opts, split='train')     #
	else:   # or use train split
		ds_train = P3D_D(src_arr, tar_arr, opts, split='train')     #



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
	models = [G, D]
	optims = [optimizerG, optimizerD]
	crits = [criterionG, criterionD]

	if opts.if_test_PA == 'n':	        # this is ont for test session (only), do train
		loss_li = []        # train loss
		err_li = []         # err test
		for epoch in range(start_epoch, opts.end_epoch_PA):
			# loop train
			loss, err, _ = loop(models, train_loader, epoch, phase='train', logger=logger, crits=crits, optims=optims, opts_in=opts)        #

			loss_li.append(loss[0])
			loss_test, err_test, _ = loop(models, test_loader, epoch, phase='test', logger=logger, crits=crits, optims=optims, opts_in=opts)  # only run G
			err_li.append(err_test)

			schedulerG.step()
			schedulerD.step()
			# can add saving point for freq, but  save the trouble it is quick
			is_best = err_test < err_best       # the mpjpe
			sv_dict = {
				'epoch': epoch + 1,  # to next epoch when start again
				'err': err_best,    # keep the best one here
				'G': G.state_dict(),
				'optim_G': optimizerG.state_dict(),
				'sch_G': schedulerG.state_dict(),
				'D': D.state_dict(),
				'optim_D': optimizerD.state_dict(),
				'sch_D': schedulerD.state_dict()
			}

			sv_pth = osp.join(model_dir, sv_hd +'.pth')
			logger.info('>>> save model to {}'.format(sv_pth))
			torch.save(sv_dict, sv_pth)
			if is_best:     # best save again
				err_best = err_test # to current one
				sv_pth = osp.join(model_dir, sv_hd + '_best.pth')
				sv_dict = {
					'epoch': epoch + 1,  # to next epoch when start again
					'err': err_test,
					'G': G.state_dict(),
					'optim_G': optimizerG.state_dict(),
					'sch_G': schedulerG.state_dict(),
					'D': D.state_dict(),
					'optim_D': optimizerD.state_dict(),
					'sch_D': schedulerD.state_dict()
				}

				logger.info('>>> save best model to {}'.format(sv_pth))
				torch.save(sv_dict, sv_pth)
		loss_arr = np.array(loss_li)
		# plt.plot(loss_arr)
		# plt.show()

		# save the loss_li for later
		if opts.if_clip_grad == 'n':
			loss_hd = sv_hd+'_clip'
		else:
			loss_hd = sv_hd
		ut_t.sv_json(rst_dir, loss_hd, loss_li, 'L')  # PA_GD/result/ScanAva_Human36M-p2_L.json

	logger.info("err best is {}".format(err_best))
	# loop final test report
	loss_test, err_test, preds = loop(models, test_loader, epoch, phase='test', logger=logger, crits=crits, optims=optims, opts_in=opts)

	if tarset_PA == 'h36m-p1':
		pth_hd = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto1'
	elif tarset_PA == 'h36m-p2':
		pth_hd = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto2'
	elif tarset_PA == 'MuCo' or tarset_PA == 'MuPoTS':      # all same set using MuPoTS
		pth_hd = 'MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test'        # MuPoTS_Btype-h36m_SBL-n_PA-n_exp_test_pred_hm.npy
	pth_hd = osp.join(opts.rst_dir, pth_hd)
	sv_pth = pth_hd +'_pred_hm_PA{}.npy'.format(opts.PA_G_mode) #     # Human36M_Btype-h36m_SBL-n_PA-n_exp_train_proto2_pred_hm_PA1.npy
	logger.info('saving hte PA test pm to {}'.format(sv_pth))
	np.save(sv_pth, preds)      #

if __name__ == '__main__':
	main()