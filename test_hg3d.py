'''
test script for hg3d (towards wild 3d human ICCV'17)
'''
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from utils.utils_pose import flip
from data import dataset
import torch.backends.cudnn as cudnn
from opt import opts, print_options, set_env
import os
import os.path as osp
from utils.logger import Colorlogger
from models.SAA import SAA
from data.dataset import genLoaderFromDs
import utils.utils_tool as ut_t
import math
import json
from models.msra_resnet import get_pose_net
from utils.config_MPPE import Config
from torch.nn.parallel.data_parallel import DataParallel
from utils.utils_tch import get_model_summary


from models.smplBL import PAnet, dct_mu, dct_std
import torch.nn as nn

def weight_init(m):     # for sbl
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

candidate_sets = ['Human36M', 'ScanAva', 'MPII', 'MSCOCO', 'MuCo', 'MuPoTS', 'SURREAL']
for i in range(len(candidate_sets)):
	exec('from data.' + candidate_sets[i] + '.' + candidate_sets[i] + ' import ' + candidate_sets[i])


# *****hg36 specific tool****
def get_preds(hm, return_conf=False):
	assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
	h = hm.shape[2]
	w = hm.shape[3]
	hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
	idx = np.argmax(hm, axis=2)

	preds = np.zeros((hm.shape[0], hm.shape[1], 2))
	for i in range(hm.shape[0]):
		for j in range(hm.shape[1]):
			preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w  # n_bch x n_jt
	if return_conf:
		conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
		return preds, conf  # bch x jt x 1  for conf , :hm
	else:
		return preds  # bch x ch x 2  for all x,ys

def get_preds_3d(heatmap, depthmap):
	# original everything to -1 ,1 here we map it to hm
	output_res = max(heatmap.shape[2], heatmap.shape[3])  # resolution?
	preds = get_preds(heatmap).astype(np.int32)  #
	preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
	for i in range(preds.shape[0]):
		for j in range(preds.shape[1]):
			idx = min(j, depthmap.shape[1] - 1)  # no exceed existing depth ch
			pt = preds[i, j]
			preds_3d[i, j, 2] = (depthmap[i, idx, pt[1], pt[0]]+1)/2  # -1 to 1 assume
			preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
	return preds_3d  # normalized x,y plus root centered depth,  bch x n_jt


h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
mpii_to_h36m = [6, 2, 1, 0, 3, 4, 5, \
                     7, 8, 9, 9, \
                     13, 14, 15, 12, 11, 10]  # spine, neck, head site, head 9 also for site

def testLoop(model, ds_test, opts={}, logger_test=None, if_svEval=False, if_svVis=False):
	'''
	generate loader from ds, then test the model, logger record, and image save directly here.
	:param model: the trained model
	:param ds_test: the test dataset
	:param logger_test: to record continuously
	:return:
	'''
	# ds, loader, iterator = dataset.genDsLoader(opts, mode=mode)

	ds_adp, loader, iter_test = genLoaderFromDs(ds_test, opts=opts)  # will be in batch format, sequenced
	preds = []
	targets = []

	# sbl model only for test so I put it here
	# dct_smplBL = {'n':0, 'cam':1, 'patch':2}
	# id_smplBL = dct_smplBL[opts.smplBL]
	smplBL = opts.smplBL
	id_smplBL = 0
	if 'cam' in smplBL:     # from cam
		id_smplBL = 2
	elif 'n' == smplBL:
		id_smplBL = 0
	else:       # from hm
		id_smplBL =1

	if id_smplBL>0 :
		fd_sbl = '/scratch/liu.shu/codesPool/3d_pose_baseline_pytorch/checkpoint'       # train in another  ,load in here
		pth_sbl = osp.join(fd_sbl, opts.smplBL, 'ckpt_last.pth.tar')
		sbl = PAnet()           # pose adaptation net.
		sbl = sbl.cuda()
		sbl.apply(weight_init)
		ckpt = torch.load(pth_sbl)
		sbl.load_state_dict(ckpt['state_dict'])
		sbl.eval()
		mu = np.array(dct_mu[opts.smplBL])
		std = np.array(dct_std[opts.smplBL])


	itr_per_epoch = math.ceil(
		ds_adp.__len__() / opts.num_gpus / opts.batch_size)  # single ds test on batch share
	if opts.testIter > 0:
		itr_per_epoch = min(itr_per_epoch, opts.testIter)
	model.eval()
	# get summary
	input_t, target_t = ds_adp[0]
	img_patch = input_t['img_patch'].unsqueeze(0)
	# writer_dict['writer'].add_graph(model, (dump_input, ))      # can't work under 1.4 get rid of
	logger_test.info(get_model_summary(model, img_patch))  # get output,  save compare ori, hmb, inc (%)

	# if load,
	tch_mean = torch.tensor(opts.pixel_mean).view([1,-1,1,1])
	tch_std = torch.tensor(opts.pixel_std).view([1,-1,1,1])


	with torch.no_grad():
		for i in tqdm(range(itr_per_epoch), desc='testing {} partition {}...'.format(opts.testset, opts.test_par)):        # loop all test
			input, target = next(iter_test)
			if id_smplBL == 1:
				pass # do the forwarding simple baseline directly get coord
			else:       # for mode 0 and 2, ran main model.
				# specific for the  hg3d
				# out = model(input['img_patch'])[-1]       # why -1 the last stack, no such thing in model
				img_patch = input['img_patch']
				img_patch = ((img_patch*tch_std+tch_mean)/255.0 - tch_mean)/tch_std
				img_patch = img_patch[:,[2,1,0]]
				# [:, [2,1,0]]      # change channel order to BGR
				# print('img_patch 0 range', img_patch[0].min(), img_patch[0].max())
				out = model(img_patch)[-1]
				pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
				                       out['depth'].detach().cpu().numpy())  # 0~1
				coord_out_mpii = pred_3d*opts.output_shape[0]        # 64

				coord_out = coord_out_mpii[:, mpii_to_h36m]   # n_bch, axis 1 size 3
				if opts.flip_test:
					flipped_input_img = flip(img_patch, dims=3)
					out_f = model(flipped_input_img)[-1]     # original direct pred3d
					pred_3d_f = get_preds_3d(out_f['hm'].detach().cpu().numpy(),
					                         out_f['depth'].detach().cpu().numpy())  # 0~1
					coord_out_mpii_f = pred_3d_f * opts.output_shape[0]  # to 64
					flipped_coord_out = coord_out_mpii_f[:, mpii_to_h36m] # n_bch
					# flipped_coord_out = flipped_coord_out[:, :-1, :]

					flipped_coord_out[:, :, 0] = opts.output_shape[1] - flipped_coord_out[:, :, 0] - 1  # flip x coordinates
					for pair in opts.ref_flip_pairs:
						flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1], :].copy(), flipped_coord_out[:, pair[0],:].copy()        # original clone

					coord_out = (coord_out + flipped_coord_out) / 2.

				# update the missing point
				coord_out[:,7] = (coord_out[:, 0] + coord_out[:, 8]) / 2    #  torso
				coord_out[:, 9] = (coord_out[:, 8] + coord_out[:, 10]) / 2   # site

				if 2 == id_smplBL: # change coordinate  if 2 nd method
					n_bch = len(coord_out)  # the batch size
					shp_out = coord_out.shape       # for recover later
					inp_smpl = (coord_out-mu)/std    # norm
					inp_smpl = inp_smpl[:,:,:2]    # n_bc x n_jt x 3 get in
					inp_smpl = inp_smpl.reshape(n_bch, -1)     # n_bch x 51
					out_smpl = sbl(torch.from_numpy(inp_smpl).float().cuda())   # estimate
					out_smpl = out_smpl.cpu().numpy()       # np
					out_smpl = out_smpl.reshape(shp_out)    # reshape
					out_smpl = out_smpl*std + mu # recover
					if False:
						print('befor change', coord_out[0])
						print('after change', out_smpl[0])
					coord_out = out_smpl    # updating


			preds.append(coord_out)  # add result
			targets.append(target['joint_hm'].cpu().numpy())    # to numpy

			if if_svVis and 0 == i % opts.svVis_step:   # every 10 step.  batch 300
				# save visuals  only 1st one in batch
				sv_dir = opts.vis_test_dir  # exp/vis/Human36M
				if opts.testset == 'Human36M':
					sv_dir += '_p{}'.format(opts.h36mProto)     # add proto to it
				img_patch_vis = ut_t.ts2cv2(input['img_patch'][0])
				idx_test = i * opts.batch_size
				skels_idx = opts.ref_skels_idx
				# get pred2d_patch
				pred2d_patch = np.zeros((3, opts.ref_joints_num))  # 3xn_jt format
				pred2d_patch[:2, :] = coord_out[0, :, :2].transpose(1, 0) / opts.output_shape[0] * opts.input_shape[
					0]  #  3 * n_jt  set depth 1 , from coord_out 0 !!
				pred2d_patch[2, :] = 1

				ut_t.save_2d_tg3d(img_patch_vis, pred2d_patch, skels_idx, sv_dir,
				                  idx=idx_test)  # make sub dir if needed, recover to test set index by indexing. good

	preds = np.concatenate(preds, axis=0)  # x,y,z :HM preds = np.concatenate(preds, axis=0)   # x,y,z :HM  into vertical long one
	targets = np.concatenate(targets, axis=0)
	if isinstance(ds_test, Human36M):  # split_flip-[y|n]_e{epoch}_[proto1/2]
		pth_hd = osp.join(opts.rst_dir, '_'.join([opts.nmTest, ds_test.data_split, 'proto' + str(ds_test.protocol)]))  #
	else:  # if not no protocol     nm: Human36M_Btype-h36m_SBL-n_PA-n_test_flip_y_proto2
		pth_hd = osp.join(opts.rst_dir, '_'.join([opts.nmTest, ds_test.data_split]))

	if if_svEval:       # save result for quick test
		pred_pth = '_'.join([pth_hd, 'pred_hm.npy'])        # eg: Human36M_Btype-h36m_SBL-n_PA-n_exp_train_proto1_pred_hm.npy
		pred_json = {'pred':preds.tolist(), 'gt':targets.tolist()}
		np.save(pred_pth, preds)        # save middle preds result
		ut_t.sv_json(opts.rst_dir, pth_hd, pred_json, 'pred_hm')    # will to json

	# if 'test' in ds_test.data_split:    # not eval large train data   all evaluated
	ds_test.evaluate(preds, jt_adj=opts.adj, logger_test=logger_test, if_svVis=if_svVis, if_svEval=if_svEval, pth_hd=pth_hd)  # shoulddn't return, as different set different metric, some only save


def main():
	# opts.model = 'MPPE' # force to mppe
	set_env(opts)   # put arg here
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	cudnn.fastest = True
	cudnn.benchmark = True
	cudnn.deterministic = False  # result unchanged
	cudnn.enabled = True
	print('---test begins---')
	print_options(opts)  # show options
	# test Logger here
	logger_testFinal = Colorlogger(opts.log_dir, 'testFinal_logs.txt')
	model_path = os.path.join(opts.model_dir, 'fusion_3d_var.pth')        # 21 jt actually
	assert os.path.exists(model_path), 'Cannot find model at ' + model_path
	logger_testFinal.info('Load checkpoint from {}'.format(model_path))
	heads = {'hm':16, 'depth':16}
	model = get_pose_net(50, heads)  # the thorax, 3 more!
	# model = DataParallel(model).cuda()    #  hg3d no parallel
	ckpt = torch.load(model_path)   # then parallel
	if type(ckpt) == type({}):
		state_dict = ckpt['state_dict']     # dict or the structure
	else:
		state_dict = ckpt.state_dict()
	model.load_state_dict(state_dict)  # final_layer.weight  copy 1152 current 1088  , final loaded in is 18
	model.eval()
	model = DataParallel(model).cuda()  # hg3d no parallel , to para



	ds_test = eval(opts.testset)(opts.test_par, opts=opts)  # can test whatever set # (h36m, test)
	if 'n' == opts.if_loadPreds:
		if_testLoad = False      # no GPU needed in this mode
	else:
		if_testLoad = True
	if if_testLoad: # load or go loop
		if isinstance(ds_test, Human36M):  # split_flip-[y|n]_e{epoch}_[proto1/2]
			pth_hd = osp.join(opts.rst_dir,'_'.join([opts.nmTest, ds_test.data_split, 'proto' + str(ds_test.protocol)]))  #
		else:  # if not no protocol     nm: Human36M_Btype-h36m_SBL-n_PA-n_test_flip_y_proto2
			pth_hd = osp.join(opts.rst_dir, '_'.join([opts.nmTest, ds_test.data_split]))
		pred_pth = '_'.join([pth_hd, 'pred_hm.npy'])
		if opts.if_hm_PA =='y': #
			pred_pth = '_'.join([pth_hd, 'pred_hm_PA{}.npy'.format(opts.PA_G_mode)])  # get the PA predicts
		logger_testFinal.info("loading test result from {}".format(pred_pth))
		preds = np.load(pred_pth)
		ds_test.evaluate(preds, jt_adj=opts.adj, logger_test=logger_testFinal, if_svVis=True, if_svEval=True, pth_hd=pth_hd)
	else:       # env issue,  models earlier otherwise,  env error
		if_svEval = True        # all saved
		testLoop(model, ds_test, opts=opts, logger_test=logger_testFinal, if_svEval=if_svEval, if_svVis=True)


if __name__ == "__main__":
	main()
