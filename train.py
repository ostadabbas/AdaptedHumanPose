import argparse
# from config import cfg
import torch
# from base import Trainer
import torch.backends.cudnn as cudnn
import os
from opt import opts, set_env
from models import modelTG
from utils.timer import Timer
import test

set_env(opts)       # set and build folder only once

def main():
	# cfg.set_args(args.gpu_ids, args.continue_train)
	gpu_ids_str = [str(i) for i in opts.gpu_ids]
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
	print('>>> Using GPU: {}'.format(gpu_ids_str))
	cudnn.fastest = True
	cudnn.benchmark = True
	# set env gpu


	# make dtLoader for train and test, testInLoop make them to iter list
	# itr_per_epoch = math.ceil(trainset_loader[0].__len__() / cfg.num_gpus / (cfg.batch_size // len(cfg.trainset))) # get the longest one

	# make models with opt specific
	model = None # todo ---
	# trainer = Trainer()
	# trainer._make_batch_generator(ds_dir=cfg.ds_dir)
	# trainer._make_model()
	# total timer (tot_timer), gpu_timer and reader_timer
	# generate loggerTrain, loggerTest,
	tot_timer = Timer()
	gpu_timer = Timer()
	read_timer = Timer()

	# loop epoch
	if opts.epoch_step > 0:
		end_epoch = min(opts.end_epoch, opts.start_epoch + opts.epoch_step)

	for epoch in range(opts.start_epoch, end_epoch):

		# model to train model
		# update model lr or by scheduler
		# set timers
		tot_timer.tic()
		read_timer.tic()
		# check set iter

		if opts.trainIter > 0:
			iter_per_epoch = min(opts.trainIter, iter_per_epoch)

		for itr in range(itr_per_epoch):  #
			# init list, iter bach concate from all set (add if_SYN)
			input_img_list, joint_img_list, joint_vis_list, joints_have_depth_list = [], [], [], []
			if_SYN_list = []
			for i in range(len(cfg.trainset)):  # loop set
				try:
					input_img, joint_img, joint_vis, joints_have_depth = next(trainer.iterator[i])  # iter b
				except StopIteration:
					trainer.iterator[i] = iter(trainer.batch_generator[i])  # set again suppose to be set already
					input_img, joint_img, joint_vis, joints_have_depth, if_SYN = next(trainer.iterator[i])

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

			# shuffle items from different datasets
			rand_idx = []
			for i in range(len(opts.trainset)):
				rand_idx.append(torch.arange(i, input_img.shape[0], len(
					cfg.trainset)))  # len(trainSet) interval 0,2,4,....then [1,3,5...]  not necessary batch is not input channel actually
			rand_idx = torch.cat(rand_idx, dim=0)
			rand_idx = rand_idx[torch.randperm(input_img.shape[0])]
			input_img = input_img[rand_idx];
			joint_img = joint_img[rand_idx];
			joint_vis = joint_vis[rand_idx];
			joints_have_depth = joints_have_depth[rand_idx];
			if_SYNs = if_SYNs[rand_idx]

			target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth, 'if_SYNs': if_SYNs}

			read_timer.toc()
			gpu_timer.tic()

			# train process
			# model.set_input, optimizer parameter

			gpu_timer.toc()

			# get all losses from model L_G, L_D, L_total, from model


			# form log infor, log it
			screen = [
				'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
				'lr: %g' % (trainer.get_lr()),
				'speed: %.2f(%.2fs r%.2f)s/itr' % (
					trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
				'%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
				'%s: %.4f' % ('loss_coord', loss_coord.detach()),
			]
			trainer.logger.info(' '.join(screen))
			# check all timer infor
			tot_timer.toc()
			tot_timer.tic()
			read_timer.tic()

			# if if_display:  update loss on visdom, visualization is OK as this is main regression. Number speaks for itself.  visdom one not urgent or necessary as we don't show image yet.

		# end of epoch, test in loop
		# model.eval()
		# loop testLoopLoader(),
		# testInLoop, test(ds_testInLoop, model): form loader, then evaluate through, no save
		# return all metric values, MPJPE, pckh, auc
		# for logger infor,  display, and logger_test save infor

		# save every epoch, save step
		if 0 == epoch % opts.save_step:
			trainer.save_model({
				'epoch': epoch,
				'network': trainer.model.state_dict(),
				'optimizer': trainer.optimizer.state_dict(),
			}, epoch)

	# after all training,  run test.test(ds_test, model) : build loader and loop through
	test.testLoop(model, mode=1, if_svRst=True, if_svVis=True)
	# save final model  wtih epoch nuber


if __name__ == "__main__":
	main()
