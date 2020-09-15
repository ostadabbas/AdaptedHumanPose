import argparse
# from config import cfg
import torch
# from base import Trainer
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from opt import opts, set_env
from models.modelTG import TaskGenNet
from utils.timer import Timer
import test
from data.dataset import genDsLoader
import math
from utils.logger import Colorlogger
from utils.vis import Visualizer, vis_keypoints, vis_3d_skeleton
from test import testLoop
import pickle
import numpy as np
import utils.utils_tool as ut_t
import cv2

# get all sets name
candidate_sets = ['Human36M', 'ScanAva', 'MPII', 'MSCOCO', 'MuCo', 'MuPoTS']
for i in range(len(candidate_sets)):
	exec('from data.' + candidate_sets[i] + '.' + candidate_sets[i] + ' import ' + candidate_sets[i])

def main():
	set_env(opts)  # set and build folder only once
	# cfg.set_args(args.gpu_ids, args.continue_train)
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	print('saving result to {}'.format(opts.exp_dir))
	cudnn.fastest = True
	cudnn.benchmark = True
	# clean up the cache
	torch.cuda.empty_cache()

	# logger
	logger_train = Colorlogger(opts.log_dir, 'train_logs.txt')
	logger_test = Colorlogger(opts.log_dir, 'test_logs.txt')
	logger_testFinal = Colorlogger(opts.log_dir, 'logger_testFinal.txt')
	logger_debug = Colorlogger(opts.log_dir, 'debug_logs.txt')

	# make dtLoader for train and gen a test set instance
	adpDs_train_li, loader_train_li, iter_train_li = genDsLoader(opts, mode=0)  # train loader
	ds_test = eval(opts.testset)("test", opts=opts) # keep a test set default h36m
	ds_testInLoop = eval(opts.testset)("testInLoop", opts=opts) # keep a test set, ds load reduce the size

	itr_per_epoch = math.ceil(
		adpDs_train_li[0].__len__() / opts.num_gpus / (opts.batch_size // len(opts.trainset)))  # get the longest one
	if opts.trainIter > 0:
		# itr_per_epoch = min(itr_per_epoch, opts.trainIter)    # smaller one
		itr_per_epoch =opts.trainIter       # hard set it

	# make models with opt specific
	model = TaskGenNet(opts) # with initialization already, already GPU-par
	if 0 == opts.start_epoch and 'y' != opts.if_scraG:
		logger_train.info('loading pretrained backbone G')
		model.load_bb_pretrain()  # init backbone
	elif opts.start_epoch > 0:      # load the epoch model
		logger_train.info('loading model from epoch {}'.format(opts.start_epoch-1))
		model.load_networks(opts.start_epoch-1)

	# init visualizer
	if opts.display_id > 0:
		visualizer = Visualizer(opts)  # only plot losses here, a loss log comes with it,
		if opts.start_epoch > 0:    # recovery
			logger_train.info('load visualizer state from epoch {}'.format(opts.start_epoch-1))
			visualizer.load(opts.start_epoch-1)

	# trainer = Trainer()
	# trainer._make_batch_generator(ds_dir=cfg.ds_dir)
	# trainer._make_model()
	# total timer (tot_timer), gpu_timer and reader_timer
	# generate loggerTrain, loggerTest,
	tot_timer = Timer()
	gpu_timer = Timer()
	read_timer = Timer()
	epoch_timer = Timer()

	total_iters = 0
	# loop epoch
	if opts.epoch_step > 0:
		end_epoch = min(opts.end_epoch, opts.start_epoch + opts.epoch_step)
	else:
		end_epoch = opts.end_epoch
	epoch = opts.start_epoch        # initalize in case no loop at all

	for epoch in range(opts.start_epoch, end_epoch):
		# update all states, if regress depth ...
		epoch_timer.tic()
		if epoch >= opts.epoch_regZ:
			model.if_z = 1.     # regress z then
			if 'y' == opts.if_fixG:
				model.fix_G()   # D will be disabled
		if opts.display_id > 0:
			visualizer.reset()
		model.train()
		# set timers
		for itr in range(itr_per_epoch):  #
			# init list, iter bach concate from all set (add if_SYN)
			input_img_list, joint_img_list, joint_vis_list, joints_have_depth_list = [], [], [], []
			wts_list = []
			if_SYN_list = []
			total_iters += 1
			tot_timer.tic()
			read_timer.tic()

			for i in range(len(opts.trainset)):  # loop set each iter combine the data
				try:
					# input_img, joint_img, joint_vis, joints_have_depth, if_SYNs = next(iter_train_li[i])  # iter b
					input, target = next(iter_train_li[i])
				except StopIteration:
					# trainer.iterator[i] = iter(trainer.batch_generator[i])  # set again suppose to be set already
					iter_train_li[i] = iter(loader_train_li[i])  # set again suppose to be set already
					# input_img, joint_img, joint_vis, joints_have_depth, if_SYN = next(iter_train_li[i])
					input, target = next(iter_train_li[i])
				# decomposed dict for concat
				input_img = input['img_patch']
				joint_img = target['joint_hm']
				joint_vis = target['vis']
				joints_have_depth = target['if_depth_v']
				if_SYN = target['if_SYN_v']
				# li2_gs_in = target['li2_gs']
				wts = target['wts_D']

				input_img_list.append(input_img)
				joint_img_list.append(joint_img)
				joint_vis_list.append(joint_vis)
				joints_have_depth_list.append(joints_have_depth)
				if_SYN_list.append(if_SYN)
				wts_list.append(wts)
				# update li2_gs
				# for i in range(opts.n_stg_D):
				# 	for j in range(opts.ref_joints_num):
				# 		li2_gs_clct[i][j].append(li2_gs_in[i][j])

			# aggregate items from different datasets into one single batch
			input_img = torch.cat(input_img_list, dim=0)
			joint_img = torch.cat(joint_img_list, dim=0)
			joint_vis = torch.cat(joint_vis_list, dim=0)
			joints_have_depth = torch.cat(joints_have_depth_list, dim=0)
			if_SYNs = torch.cat(if_SYN_list, dim=0)
			wts = torch.cat(wts_list, dim=0)

			# for i in range(opts.n_stg_D):
			# 	for j in range(opts.ref_joints_num):
			# 		li2_gs_clct[i][j] = torch.cat(li2_gs_clct[i][j], dim=0)

			# shuffle items from different datasets, one batch is uselss...whatever
			rand_idx = []
			for i in range(len(opts.trainset)):
				rand_idx.append(torch.arange(i, input_img.shape[0], len(
					opts.trainset)))  # len(trainSet) interval 0,2,4,....then [1,3,5...]  not necessary batch is not input channel actually, first of each set next to each other
			rand_idx = torch.cat(rand_idx, dim=0)
			rand_idx = rand_idx[torch.randperm(input_img.shape[0])]
			input_img = input_img[rand_idx];
			joint_img = joint_img[rand_idx];
			joint_vis = joint_vis[rand_idx];
			joints_have_depth = joints_have_depth[rand_idx];
			if_SYNs = if_SYNs[rand_idx]
			wts = wts[rand_idx]
			# for i in range(opts.n_stg_D):
			# 	for j in range(opts.ref_joints_num):
			# 		li2_gs_clct[i][j] = li2_gs_clct[i][j][rand_idx]

			# recompose for input
			if False:
				print(wts[0].sum())     #  check wts
			input = {'img_patch': input_img}     # dict input , target for train
			# target = {'joint_hm': joint_img, 'vis': joint_vis, 'if_depth_v': joints_have_depth, 'if_SYN_v': if_SYNs, 'li2_gs':li2_gs_clct}
			target = {'joint_hm': joint_img, 'vis': joint_vis, 'if_depth_v': joints_have_depth, 'if_SYN_v': if_SYNs, 'wts_D': wts}

			rd_tm = read_timer.toc()
			gpu_timer.tic()

			if opts.ifb_debug:
				logger_debug.info('input patch size', input_img.shape)
				logger_debug.info('joint_hm 0')
				logger_debug.info(joint_img[0])     # first in batch
				logger_debug.info('vis', joint_vis)
				logger_debug.info('if_depth_v', joints_have_depth)
				logger_debug.info('if_SYN_v', if_SYNs)

			# train process
			model.set_input(input, target)
			model.optimize_parameters()
			if itr == 0 and True:        #  for mem test
				print("test mem usage")
				print(ut_t.get_gpu_memory_map())

			gpu_tm = gpu_timer.toc()
			# get all losses from model L_G, L_D, L_total, from model
			losses = model.get_current_losses()     # T, G_GAN, D, -1 for invalid
			loss_str_li = []
			for k,v in losses.items():
				loss_str_li.append('%s:%.4f' % (k, v))
			loss_str ='loss:' + ' '.join(loss_str_li)

			tot_tm = tot_timer.toc()
			screen = [
				'Epoch %d/%d itr %d/%d:' % (epoch, end_epoch, itr, itr_per_epoch),
				'speed: %.2f(r%.2f g%.2f)s/itr' % (
					tot_tm, rd_tm, gpu_tm),
				'%.2fh/epoch' % (tot_tm / 3600. * itr_per_epoch),
				loss_str,
			]
			logger_train.info(' '.join(screen))
			if opts.display_id > 0 and total_iters % opts.update_html_freq == 0:
				idx_vis = 0
				skels_idx = opts.ref_skels_idx
				# get joint 2d, 3d, img patch, plot . what format needed?
				img_vis = input['img_patch'][0]
				img_vis = ut_t.ts2cv2(img_vis)
				coord_out = model.coord[idx_vis].cpu().detach().numpy()
				# 2d img
				pred2d_patch = np.zeros((3, opts.ref_joints_num))  # 3xn_jt format
				pred2d_patch[:2, :] = coord_out[:, :2].transpose(1, 0) / opts.output_shape[0] * opts.input_shape[0]  # x * n_jt ?
				pred2d_patch[2, :] = 1
				img_2d = vis_keypoints(img_vis, pred2d_patch, skels_idx)    # cv2
				# 3d img
				nm_3d = osp.join(opts.vis_dir, 'tmp.jpg')
				rg = [[0, opts.output_shape[1]], [0, opts.output_shape[0]], [0, opts.depth_dim]]
				ut_t.vis_3d(coord_out, skels_idx, sv_pth=nm_3d, rg=rg)    # avoid conflicts with other test in own vis_dir    # plt
				shp = img_2d.shape[:2]  # h, w
				img_3d = cv2.resize(cv2.imread(nm_3d), (shp[1], shp[0]))
				vis_dict={'2d': img_2d[:, :, ::-1], '3d': img_3d[:, :, ::-1]}     # use the skimage RGB, seems visome likes that
				visualizer.display_current_results(vis_dict, epoch, False)
				visualizer.plot_current_losses(epoch, float(itr)/itr_per_epoch, losses)
		# test in loop
		logger_test.info('testInLoop Epoch {}'.format(epoch))
		err_dict = testLoop(model, ds_testInLoop, opts=opts, logger_test=logger_test) # loop, preds, eval, print and vis
		if type(err_dict) == dict:
			# eval_dict = {'MPJPE PA': err_dict['p1_err'], 'MPJPE': err_dict['p2_err']}
			eval_dict = {'MPJPE': err_dict['p2_err']} # don't user MPJPE PA , unstable during train
			if opts.display_id > 0:
				visualizer.plot_metrics(epoch, eval_dict)

		# save every epoch, save step
		if 0 == epoch % opts.save_step and epoch > 0:   # don't save first for easy debug
			model.save_networks(epoch)  # all model optim dict saved
			# visualizer can't pickle due to local vis item
			if opts.display_id > 0:
				vis_dict = {}
				sv_keys = ['plot_data', 'evals']
				for key in sv_keys:
					vis_dict[key] = getattr(visualizer, key, None)  # at least None is saved?
				np.save(osp.join(opts.vis_dir, 'vis_{}.npy'.format(epoch)), vis_dict)
		# model to train, update learning rate
		model.train()
		model.update_learning_rate()        # update at the end
		logger_train.info('epoch actual total time {}'.format(epoch_timer.toc()))

	# final save for model and visualizer both
	if end_epoch > opts.start_epoch:  # finally if end>start, means some training really happens
		model.save_networks(epoch)  #
		if opts.display_id > 0:
			vis_dict = {}
			sv_keys = ['plot_data', 'evals']        # save the visualizer data
			for key in sv_keys:
				vis_dict[key] = getattr(visualizer, key, None)  # at least None is saved?
			np.save(osp.join(opts.vis_dir, 'vis_{}.npy'.format(epoch)), vis_dict)

	# after all training,  run test.test(ds_test, model) : build loader and loop through
	if epoch >= opts.end_epoch-1 and 'y' == opts.if_finalTest:  # only to final epoch ,do this test.
		logger_test.info('perform final test')
		test.testLoop(model, ds_test, opts=opts, logger_test=logger_testFinal, if_svEval=True, if_svVis=True)
	# save final model  wtih epoch nuber


if __name__ == "__main__":
	main()
