'''
Parse arguments for training. act as config
'''
import os
import os.path as osp
import sys
import argparse
import glob
import utils.utils_tool as tool
import utils.utils_pose as ut_p
import json
# if don't want to install put add cocoapi path here
# coco_api_dir = r'G:\My Drive\research\Sarah Ostadabbas\codePool\cocoapi\PythonAPI'
# tool.add_pypath(coco_api_dir)     # uncom for deploy
from data.Human36M.Human36M import Human36M
from data.MPII.MPII import MPII




def parseArgs():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# add argument in
	#-- env settings
	parser.add_argument('--ds_dir', default='/scratch/liu.shu/datasets', help='dataset directionry')
	parser.add_argument('--output_dir', default='output', help='default output dirs') # model, rst, vis will be under this dir.
	# parser.add_argument('--cocoapi_dir', default='/scratch/liu.shu/codesPool/cocoapi/PythonAPI', help='coco api dir')
	parser.add_argument('--cocoapi_dir', default=r'G:\My Drive\research\Sarah Ostadabbas\codePool\cocoapi\PythonAPI', help='coco api dir')
	parser.add_argument('--ifb_debug', action='store_true')
	parser.add_argument('--suffix_ptn_train', default='{net_BB}_{if_scraG}scraG_{lmd_D}D{n_layers_D}_{if_ylB}yl_{rt_SYN}rtSYN_regZ{epoch_regZ}_{if_fixG}fG_{if_normBone}nmBone', help='the suffix pattern to form name') ## --
	parser.add_argument('--suffix_exp_train', default='exp', help='the manually given suffix for specific test')
	parser.add_argument('--suffix_ptn_test', default='{testset}_{if_flipTest}flip_{if_gtRtTest}GtRt_{if_adj}Adj.json', help='the suffix pattern to form name')
	parser.add_argument('--suffix_exp_test', default='exp', help='mannualy added suffix for test result')
	# -- ds specific
	parser.add_argument('--h36mProto', default=2)

	# -- train setting
	parser.add_argument('--trainset', nargs='+', default=['ScanAva'], help='give the main ds here the iter number will follow this one')
	# parser.add_argument('--if_D', default='y', help='if use discriminator, if single ds, then automatically set to n')
	parser.add_argument('--lmd_D', default=0.01, type=float, help='weight for D loss, 0. for no D')
	parser.add_argument('--if_ylB', default='y', help='if train the low task in target B domain')
	parser.add_argument('--if_tightBB_ScanAva', default='n', help='if use tight bb for scanAva')
	parser.add_argument('--rt_SYN', default=1., type=float, help='the ratio of the subjects used in training, either scalAvan or SURREAL with floor operation, make it gerater than one I think')
	parser.add_argument('--inp_sz', default=256, help='input image size')
	parser.add_argument('--end_epoch', default=25, type=int, help='when reach this epoch, will stop. python index style, your model will be saved as epoch_tar-1')
	parser.add_argument('--epoch_step', default=5, type=int, help='mainly for time constrained system, each time only train step epoches, -1 for all')
	parser.add_argument('--trainIter', default=-1, type=int, help='train iters each epoch, -1 for whole set. For debug purpose (DBP)')
	parser.add_argument('--if_normBone', default='y', help='true of false [y|n] normalized the bones')
	parser.add_argument('--if_fixG', default='n', help='if fix G after introducing z training')
	parser.add_argument('--epoch_regZ', default=0, help='when to start training z part. depend on option to see if fixG or not. 0 means from very beginning')
	parser.add_argument('--optimizer', default='nadam', help='[adam|nadam]')
	parser.add_argument('--lr', default=1e-3)
	parser.add_argument('--lr_policy', default='multi_step', help='[step|plateau|multi_step|cosine]')
	parser.add_argument('--lr_dec_epoch', nargs='+', type=int, default=[17, 21], help='the lr decay epoch, each time by decay factor')   # form string sec dec17-21 if needed
	# parser.add_argument('--lr_dec_factor', default=10.0, type=float)
	parser.add_argument('--batch_size', default=32, type=int)
	# test batch size 16 what is the reason,, no idea
	parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='the ids of the gpu')
	# parser.add_argument('--if_coninue', default='y', help='if continue to train')
	parser.add_argument('--start_epoch', default=-1, type=int, help='where to being the epoch, -1 for continue, others hard indicating. For safety, if start epoch is the lastest as saved model, will quit ')
	parser.add_argument('--if_scraG', default='y', help='if backbone net from scratch')
	parser.add_argument('--init_type', default='xavier', help='weight initialization mode, gain 0.02 fixed in')
	parser.add_argument('--n_thread', default=20, help='how many threads')
	parser.add_argument('--save_step', default=5, help='how many steps to save model')
	parser.add_argument('--svVis_step', default=5, help='step to save visuals')
	parser.add_argument('--if_cmJoints', default='y', help='if use common joints across datasets without including the contriversal parts like sites and upper neck, torso, but only neck and head, pelvis will be always kept')

	# -- visualization
	parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
	parser.add_argument('--display_server', type=str, default="http://localhost",
	                    help='visdom server of the web display')
	parser.add_argument('--display_env', type=str, default='main',
	                    help='visdom display environment name (default is "main")')
	parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
	parser.add_argument('--use_html', action='store_true', help='if use html')
	parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

	# -- test setting
	parser.add_argument('--testset', default='Human36M', help='testset, usually single')
	parser.add_argument('--testIter', type=int, default=-1, help='test iterations final and epoch test, -1 for all, DBP')
	# parser.add_argument('--n_foldingLpTest', type=int, default=20, help='downsample the epoch test to quicken the in loop test process. 1 for full test in loop')
	parser.add_argument('--gan_mode', default='lsgan', help='gan type [lsgan|vanilla]')
	parser.add_argument('--if_flipTest', default='n')
	parser.add_argument('--if_gtRtTest', default='y', help='if use gt distance for root')
	parser.add_argument('--if_adj', default='y', help='if adjust the root location to adapt different dataset')
	parser.add_argument('--if_aveBoneRec', default='y', help='if use average boneLento recover the estimation')

	# -- network settings
	parser.add_argument('--n_layers_D', type=int, default=2, help='descriminator layer number, for 8 bb, 2 layers are good')
	parser.add_argument('--net_BB', default='res50', help='backbone net type [res50|res101|res152], can extended to VGG different layers, not so important , add later')

	# hardwired parameters
	opts = parser.parse_args()   # all cmd infor

	opts.input_shape = (opts.inp_sz, opts.inp_sz)  # tuple size
	opts.output_shape = (opts.input_shape[0]//4, opts.input_shape[1]//4)
	opts.depth_dim = opts.input_shape[0]//4       # save as output shape, df 64.
	opts.bbox_3d_shape = (2000, 2000, 2000)  # depth, height, width
	opts.pixel_mean = (0.485, 0.456, 0.406)  # perhaps for RGB normalization  after divide by 255
	opts.pixel_std = (0.229, 0.224, 0.225)

	opts.ref_joints_name = Human36M.joints_name     # stick to Human36M, we can not evaluate but keep all
	opts.ref_flip_pairs_name = Human36M.flip_pairs_name
	# post result
	opts.ref_joints_num = len(opts.ref_joints_name)  # how image output
	opts.ref_flip_pairs =ut_p.nameToIdx(opts.ref_flip_pairs_name, opts.ref_joints_name)
	opts.ref_root_idx = opts.ref_joints_name.index('Pelvis')
	# option1 for h36m joints
	if 'y' != opts.if_cmJoints:
		opts.ref_skels_name = Human36M.skels_name       # draw from rediction skels
		opts.ref_evals_name = Human36M.joints_name      # if in eval, keep otherwise, drop
	else: # todo replace with ScanAva definition  for common minimum jonits
		opts.ref_skels_name = MPII.skels_name
		opts.ref_evals_name = MPII.joints_name

	opts.ref_skels_idx = ut_p.nameToIdx(opts.ref_skels_name, opts.ref_joints_name)
	opts.ref_evals_idx = ut_p.nameToIdx(opts.ref_evals_name, opts.ref_joints_name)

	opts.clipMode = '01'        # for save image purpose
	opts.adj_dict = {     # fill this if after calculation
		'Human36M': (0, 0, 0),
		'MuPoTS': (0, 0, 0)
	}
	opts.adj = opts.adj_dict[opts.testset]      # choose the one

	# Derived parameters, model, result part...
	# form exp folder
	nmT = '-'.join(opts.trainset)  # init
	suffix_train = (opts.suffix_ptn_train.format(
		**vars(opts))) if opts.suffix_ptn_train != '' else ''  # vars return __dict__ attribute
	nmT = '_'.join([nmT, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix
	opts.name = nmT     # current experiment name
	opts.exp_dir = osp.join(opts.output_dir, nmT)
	opts.nmTest = (opts.suffix_ptn_test.format(**vars(opts))) if opts.suffix_ptn_test != '' else ''
	opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
	opts.vis_dir = osp.join(opts.exp_dir, 'vis')
	opts.log_dir = osp.join(opts.exp_dir, 'log')
	opts.result_dir = osp.join(opts.exp_dir, 'result')
	opts.num_gpus = len(opts.gpu_ids)
	opts.web_dir = osp.join(opts.exp_dir, 'web')
	opts.vis_test_dir = osp.join(opts.vis_dir, opts.testset)

	yn_dict = {'y': True, 'n': False}
	opts.flip_test = yn_dict[opts.if_flipTest]
	opts.use_gt_info = yn_dict[opts.if_gtRtTest]

	# for start epoch   name  [epoch]_net_[netName].pth
	model_file_list = glob.glob(osp.join(opts.model_dir, '*.pth'))
	if model_file_list:
		cur_epoch = max([tool.getNumInStr(fNm)[0] for fNm in model_file_list])
		start_epoch_sv = cur_epoch + 1
	else:
		start_epoch_sv = 0

	if opts.start_epoch == -1:
		opts.start_epoch = start_epoch_sv
	elif start_epoch_sv != opts.start_epoch:
		print('not latest epoch, for protection concern, please clean up exp folder mannually ')
		exit(-1)

	# otherwise, do nothing use the current start_epoch
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
	from utils.utils_tool import add_pypath, make_folder
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
	make_folder(opts.web_dir)

	# for continue, load previous setting to keep model consistency
	if opts.start_epoch == 0 :
		with open(osp.join(opts.exp_dir, 'opts.json'), 'w') as f:
			json.dump(opts, f)      # save starting point opt.




