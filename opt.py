'''
Parse arguments for training. act as config
'''
import os
import os.path as osp
import sys
import argparse
import glob
import utils.tool_utils as tool

def parseArgs():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# add argument in
	#-- env settings
	parser.add_argument('--ds_dir', default='/scratch/liu.shu/datasets', help='dataset directionry')
	parser.add_argument('--output_dir', default='output', help='default output dirs') # model, rst, vis will be under this dir.
	parser.add_argument('--cocoapi_dir', default='/scratch/liu.shu/codesPool/cocoapi/PythonAPI', help='coco api dir')
	parser.add_argument('--suffix_ptn_train', default='{if_scraG}scraG_{lmd_D}D{n_layers_D}_s3d{epoch_s3d}_{if_fixG}fG_{if_normBone}nmBone', help='the suffix pattern to form name') ## --
	parser.add_argument('--suffix_exp_train', default='exp', help='the manually given suffix for specific test')
	parser.add_argument('--suffix_ptn_test', default='{testset}_{if_flipTest}flip_{if_gtRtTest}GtRt_{if_adj}Adj.json', help='the suffix pattern to form name')
	parser.add_argument('--suffix_exp_test', default='exp', help='mannualy added suffix for test result')
	parser.add_argument(action='store_true', help='if employ visdom to show result')

	# -- train setting
	parser.add_argument('--trainset', nargs='+', default=['ScanAva'])
	parser.add_argument('--if_D', default='y', help='if use discriminator')
	parser.add_argument('--if_tightBB_ScanAva', default='n', help='if use tight bb for scanAva')
	parser.add_argument('--inp_sz', default=256, help='input image size')
	parser.add_argument('--end_epoch', default=25, type=int, help='when reach this epoch, will stop. python index style, your model will be saved as epoch_tar-1')
	parser.add_argument('--epoch_step', default=5, type=int, help='mainly for time constrained system, each time only train step epoches, -1 for all')
	parser.add_argument('--trainIter', default=-1, type=int, help='train iters each epoch, -1 for whole set. For debug purpose (DBP)')
	parser.add_argument('--if_normBone', default='y', help='true of false [y|n] normalized the bones')
	parser.add_argument('--if_fixG', default='n', help='if fix G after introducing z training')
	parser.add_argument('--epoch_s3d', default=0, help='when to start training z part. depend on option to see if fixG or not. 0 means from very beginning')
	parser.add_argument('--lr', default=1e-3)
	parser.add_argument('--lr_dec_epoch', nargs='+', type=int, default=[17, 21], help='the lr decay epoch, each time by decay factor')   # form string sec dec17-21 if needed
	parser.add_argument('--lr_dec_factor', default=10.0, type=float)
	parser.add_argument('--batch_size', default=32, type=int)
	# test batch size 16 what is the reason,, no idea
	parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='the ids of the gpu')
	parser.add_argument('--if_coninue', default='y', help='if continue to train')
	parser.add_argument('--if_scraG', default='y', help='if backbone net from scratch')
	parser.add_argument('--n_thread', default=20, help='how many threads')
	parser.add_argument('--save_step', default=5, help='how many steps to save model')
	parser.add_argument('--svVis_step', default=5, help='step to save visuals')


	# -- test setting
	parser.add_argument('--testset', default='Human36M', help='testset, usually single')
	parser.add_argument('--testIter', type=int, default=-1, help='test iterations final and epoch test, -1 for all, DBP')
	# parser.add_argument('--n_foldingLpTest', type=int, default=20, help='downsample the epoch test to quicken the in loop test process. 1 for full test in loop')
	parser.add_argument('--lmd_D', default=10.0, type=float, help='weight for discriminator with task loss to be 1')
	parser.add_argument('--if_flipTest', default='n')
	parser.add_argument('--if_gtRtTest', default='y', help='if use gt distance for root')
	parser.add_argument('--if_adj', default='y', help='if adjust the root location to adapt different dataset')

	# -- network settings
	parser.add_argument('--n_layers_D', type=int, default=3, help='descriminator layer number')
	parser.add_argument('--net_BB', default='res50', help='backbone net type [res50|res101|res152|VGG|openPose]')

	# hardwired parameters
	opts = parser.parse_args()   # all cmd infor

	opts.input_shape = (opts.inp_sz, opts.inp_sz)  # tuple size
	opts.output_shape = (opts.input_shape[0]//4, opts.input_shape[1]//4)
	opts.depth_dim = opts.input_shape[0]//4       # save as output shape, df 64.
	opts.bbox_3d_shape = (2000, 2000, 2000)  # depth, height, width
	opts.pixel_mean = (0.485, 0.456, 0.406)  # perhaps for RGB normalization  after divide by 255
	opts.pixel_std = (0.229, 0.224, 0.225)
	# Human36M version joints,  you can define to the joints you like to evaluate
	opts.ref_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax') # this is for human 3D
	opts.adj_dict={     # fill this if after calculation
		'Human36M': (0, 0, 0),
		'MuPoTS': (0, 0, 0)
	}


	# Derived parameters, model, result part...
	# form exp folder
	nmT = '-'.join(opts.trainset)  # init
	suffix_train = (opts.suffix_ptn_train.format(
		**vars(opts))) if opts.suffix_ptn_train != '' else ''  # vars return __dict__ attribute
	nmT = '_'.join([nmT, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix
	opts.exp_dir = osp.join(opts.output_dir, nmT)
	opts.nmTest = (opts.suffix_ptn_test.format(**vars(opts))) if opts.suffix_ptn_test != '' else ''
	opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
	opts.vis_dir = osp.join(opts.exp_dir, 'vis')
	opts.log_dir = osp.join(opts.exp_dir, 'log')
	opts.result_dir = osp.join(opts.exp_dir, 'result')
	opts.num_gpus = len(opts.gpu_ids)

	yn_dict = {'y': True, 'n': False}
	opts.flip_test = yn_dict[opts.if_flipTest]
	opts.use_gt_info = yn_dict[opts.if_gtRtTest]

	# for start epoch   name  [epoch]_net_[netName].pth
	model_file_list = glob.glob(osp.join(opts.model_dir, '*.pth'))
	if model_file_list:
		cur_epoch = max([tool.getNumInStr(fNm)[0] for fNm in model_file_list])
		opts.start_epoch = cur_epoch + 1
	else:
		opts.start_epoch = 0

	return opts


def print_options(opt):
	"""Print and save options

	It will print both current options and default values(if different).
	It will save options into a text file / [checkpoints_dir] / opt.txt
	"""
	message = ''
	message += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
		comment = ''
		# default = self.parser.get_default(k)
		# if v != default:
		# 	comment = '\t[default: %s]' % str(default)
		message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
	message += '----------------- End -------------------'
	print(message)

	# save to the disk
	# expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
	# tool.mkdirs(expr_dir)  # all option will mk dirs
	file_name = os.path.join(opt.exp_dir, 'opts.txt')
	with open(file_name, 'wt') as opt_file:
		opt_file.write(message)
		opt_file.write('\n')

opts = parseArgs()

def set_env(opts):
	# set sys paths
	sys.path.insert(0, 'common')
	from utils.dir_utils import add_pypath, make_folder
	add_pypath(osp.join('data'))
	for i in range(len(opts.trainset)):
	    add_pypath(osp.join('data', opts.trainset[i]))
	add_pypath(opts.cocoapi_dir)     # add coco dir to it
	add_pypath(osp.join('data', opts.testset))

	# add folders
	make_folder(opts.model_dir)
	make_folder(opts.vis_dir)
	make_folder(opts.log_dir)
	make_folder(opts.result_dir)


