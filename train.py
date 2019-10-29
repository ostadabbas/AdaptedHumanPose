import argparse
# from config import cfg
import torch
# from base import Trainer
import torch.backends.cudnn as cudnn
import os
from opt import opts, set_env
from models.modelTG import TaskGenNet
from utils.timer import Timer
import test
from data.dataset import genDsLoader
import math
from utils.logger import Colorlogger
from utils.vis import Visualizer
from test import testLoop

set_env(opts)       # set and build folder only once

def main():
	# cfg.set_args(args.gpu_ids, args.continue_train)
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	cudnn.fastest = True
	cudnn.benchmark = True

	# logger
	logger_train = Colorlogger(opts.log_dir, 'train_logs.txt')
	logger_test = Colorlogger(opts.log_dir, 'test_logs.txt')
	logger_debug = Colorlogger(opts.log_dir, 'debug_logs.txt')

	# make dtLoader for train and gen a test set instance
	adpDs_train_li, loader_train_li, iter_train_li = genDsLoader(opts, mode=0)  # train loader
	ds_test = eval(opts.testset)("test", opts=opts) # keep a test set
	ds_testInLoop = eval(opts.testset)("testInLoop", opts=opts) # keep a test set

	itr_per_epoch = math.ceil(
		adpDs_train_li[0].__len__() / opts.num_gpus / (opts.batch_size // len(opts.trainset)))  # get the longest one
	if opts.trainIter > 0:
		itr_per_epoch = min(itr_per_epoch, opts.trainIter)

	# make models with opt specific
	model = TaskGenNet(opts) # with initialization already, already GPU-par
	if 0 == opts.start_epoch and 'y' == opts.if_scraG:
		model.load_bb_pretrain()  # init backbone
	elif opts.start_epoch > 0:      # load the epoch model
		logger_train.info('loading model from epoch {}'.format(opts.start_epoch-1))
		model.load_networks(opts.start_epoch-1)

	# trainer = Trainer()
	# trainer._make_batch_generator(ds_dir=cfg.ds_dir)
	# trainer._make_model()
	# total timer (tot_timer), gpu_timer and reader_timer
	# generate loggerTrain, loggerTest,
	tot_timer = Timer()
	gpu_timer = Timer()
	read_timer = Timer()

	visualizer = Visualizer(opts)   # only plot losses here, there is logger for record and print, only for current session, restart will not be saved.
	total_iters = 0

	# loop epoch
	if opts.epoch_step > 0:
		end_epoch = min(opts.end_epoch, opts.start_epoch + opts.epoch_step)

	for epoch in range(opts.start_epoch, end_epoch):
		# update all states, if regress depth ...
		if epoch >= opts.regZ:
			model.if_z = 1.     # regress z then
			if 'y' == opts.if_fixG:
				model.fix_G()   # D will be disabled
		visualizer.reset()
		model.train()

		# set timers
		tot_timer.tic()
		read_timer.tic()

		for itr in range(itr_per_epoch):  #
			# init list, iter bach concate from all set (add if_SYN)
			input_img_list, joint_img_list, joint_vis_list, joints_have_depth_list = [], [], [], []
			if_SYN_list = []
			total_iters += 1

			for i in range(len(opts.trainset)):  # loop set
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
				joint_vis = target['vs']
				joints_have_depth = target['if_depth_v']
				if_SYN = target['if_SYN_v']

				input_img_list.append(input_img)
				joint_img_list.append(joint_img)
				joint_vis_list.append(joint_vis)
				joints_have_depth_list.append(joints_have_depth)
				if_SYN_list.append(if_SYN)

			# aggregate items from different datasets into one single batch
			input_img = torch.cat(input_img_list, dim=0)
			joint_img = torch.cat(joint_img_list, dim=0)
			joint_vis = torch.cat(joint_vis_list, dim=0)
			joints_have_depth = torch.cat(joints_have_depth_list, dim=0)
			if_SYNs = torch.cat(if_SYN_list, dim=0)

			# shuffle items from different datasets, one batch is uselss...whatever
			rand_idx = []
			for i in range(len(opts.trainset)):
				rand_idx.append(torch.arange(i, input_img.shape[0], len(
					opts.trainset)))  # len(trainSet) interval 0,2,4,....then [1,3,5...]  not necessary batch is not input channel actually
			rand_idx = torch.cat(rand_idx, dim=0)
			rand_idx = rand_idx[torch.randperm(input_img.shape[0])]
			input_img = input_img[rand_idx];
			joint_img = joint_img[rand_idx];
			joint_vis = joint_vis[rand_idx];
			joints_have_depth = joints_have_depth[rand_idx];
			if_SYNs = if_SYNs[rand_idx]

			# recompose for input
			input = {'img_patch':input_img}     # dict input , target for train
			target = {'joint_hm': joint_img, 'vis': joint_vis, 'if_depth_v': joints_have_depth, 'if_SYN_v': if_SYNs}

			read_timer.toc()
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

			gpu_timer.toc()
			# get all losses from model L_G, L_D, L_total, from model
			losses = model.get_current_losses()     # T, G_GAN, D
			screen = [
				'Epoch %d/%d itr %d/%d:' % (epoch, end_epoch, itr, itr_per_epoch),
				'speed: %.2f(%.2fs r%.2f)s/itr' % (
					tot_timer.average_time, gpu_timer.average_time, read_timer.average_time),
				'%.2fh/epoch' % (tot_timer.average_time / 3600. * itr_per_epoch),
				'%s: %.4f:%.4f:%.4f' % ('loss T:G_GAN:D', losses['T'], losses['G_GAN'], losses['D']),
			]
			logger_train.info(' '.join(screen))

			if opts.display_id > 0:
				visualizer.plot_current_losses(epoch, float(itr)/itr_per_epoch, losses)
			# check all timer infor
			tot_timer.toc()
			tot_timer.tic()
			read_timer.tic()

		# end of epoch, test in loop
		# model.eval()
		# loop testLoopLoader(),
		# testInLoop, test(ds_testInLoop, model): form loader, then evaluate through, no save
		# return all metric values, MPJPE, pckh, auc
		# for logger infor,  display, and logger_test save infor
		err_dict = testLoop(model, ds_testInLoop, logger_test=logger_test) # default not save result and
		if err_dict.type == dict:
			eval_dict = {'MPJPE PA': err_dict['p1_err'], 'MPJPE': err_dict['p2_err']}
			visualizer.plot_metrics(epoch, eval_dict)

		# save every epoch, save step
		if 0 == epoch % opts.save_step:
			model.save_networks(epoch)  # all model optim dict saved

		# model to train, update learning rate
		model.train()
		model.update_learning_rate()

	# after all training,  run test.test(ds_test, model) : build loader and loop through
	test.testLoop(model, ds_test, logger_test=logger_test, if_svEval=True, if_svVis=True)
	# save final model  wtih epoch nuber


if __name__ == "__main__":
	main()
