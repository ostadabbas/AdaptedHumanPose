'''
Parse arguments for training. act as config
'''
import os
import os.path as osp
import sys
import argparse
import glob
import utils.utils_tool as ut_t
import utils.utils_pose as ut_p
import json
# if don't want to install to package put add cocoapi path here
from data.ScanAva.ScanAva import ScanAva
from data.Human36M.Human36M import Human36M

if_discovery = True # 0 thread,  ScanAva eval,  dsdir, output dir
evals_name_config = {
	'h36m':Human36M.evals_name,
	'scanava': ScanAva.evals_name,
	'cmJoints': (
	# "R_Ankle",
	"R_Knee",
	"R_Hip",
	"L_Hip",
	"L_Knee",
	# "L_Ankle",
	"R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
	"L_Elbow", "L_Wrist", "Thorax",
	# "Head",
	"Pelvis",
	"Torso",
	# "Neck"
	)
}
def parseArgs():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# add argument in
	#-- env settings
	if if_discovery:
		parser.add_argument('--ds_dir', default='/scratch/liu.shu/datasets', help='dataset directionry')      # for discovery
	else:
		parser.add_argument('--ds_dir', default=r'S:\ACLab\datasets', help='dataset directionry') # local
	# parser.add_argument('--ds_dir', default='/home/jun/SDrive/datasets', help='dataset directionry')  # for AR
	if if_discovery:
		parser.add_argument('--output_dir', default='output', help='default output dirs') # code local.
	else:
		parser.add_argument('--output_dir', default=r'S:\ACLab\rst_model\taskGen3d\output', help='default output dirs') # model, rst, vis will be under this dir.
	parser.add_argument('--cocoapi_dir', default=None, help='coco api dir. If cocoapi is not installed as a package, you can point this directly to that folder')
	parser.add_argument('--ifb_debug', action='store_true')
	parser.add_argument('--suffix_ptn_train', default='{net_BB}_{if_scraG}-scraG_{lmd_D}D{n_layers_D}_{if_ylB}-yl_{rt_SYN}rtSYN_regZ{epoch_regZ}_{if_fixG}-fG_{if_normBone}-nmBone_{optimizer}_lr{lr}', help='the suffix pattern to form name') ## --
	parser.add_argument('--suffix_exp_train', default='exp', help='the manually given suffix for specific test')
	parser.add_argument('--suffix_ptn_test', default='{testset}_flip-{if_flipTest}_GtRt-{if_gtRtTest}_Btype-{bone_type}_avB-{if_aveBoneRec}_e{start_epoch}', help='the suffix pattern to form name')
	parser.add_argument('--suffix_exp_test', default='exp', help='mannualy added suffix for test result')
	# -- ds specific
	parser.add_argument('--h36mProto', default=2, type=int)

	# -- train setting
	parser.add_argument('--trainset', nargs='+', default=['ScanAva', 'MSCOCO', 'MPII'], help='give the main ds here the iter number will follow this one')
	# parser.add_argument('--if_D', default='y', help='if use discriminator, if single ds, then automatically set to n')
	parser.add_argument('--gan_mode', default='lsgan', help='gan type [lsgan|vanilla]')
	parser.add_argument('--lmd_D', default=0.01, type=float, help='weight for D loss, 0. for no D')
	parser.add_argument('--if_ylB', default='y', help='if train the low task in target B domain')
	parser.add_argument('--if_tightBB_ScanAva', default='y', help='if use tight bb for scanAva')
	parser.add_argument('--rt_SYN', default=1, type=int, help='the ratio of the subjects used in training, larger step will make less training data')
	parser.add_argument('--inp_sz', default=256, type=int, help='input image size')
	parser.add_argument('--end_epoch', default=25, type=int, help='when reach this epoch, will stop. python index style, your model will be saved as epoch_tar-1, ori 25 ')
	parser.add_argument('--epoch_step', default=9, type=int, help='mainly for time constrained system, each time only train step epoches, -1 for all')
	parser.add_argument('--trainIter', default=-1, type=int, help='train iters each epoch, -1 for whole set. For debug purpose (DBP)')
	parser.add_argument('--if_normBone', default='n', help='true of false [y|n] normalized the bones')
	parser.add_argument('--if_fixG', default='n', help='if fix G after introducing z training')
	parser.add_argument('--epoch_regZ', default=0,  type=int, help='when to start training z part. depend on option to see if fixG or not. 0 means from very beginning')
	parser.add_argument('--optimizer', default='adam', help='[adam|nadam]')
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--lr_policy', default='multi_step', help='[step|plateau|multi_step|cosine]')
	parser.add_argument('--lr_dec_epoch', nargs='+', type=int, default=[17, 21], help='the lr decay epoch, each time by decay factor ori 17,21')   # form string sec dec17-21 if needed
	# parser.add_argument('--lr_dec_factor', default=10.0, type=float)
	parser.add_argument('--batch_size', default=30, type=int)
	# test batch size 16 what is the reason,, no idea
	parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='the ids of the gpu')
	# parser.add_argument('--if_coninue', default='y', help='if continue to train')
	parser.add_argument('--start_epoch', default=-1, type=int, help='where to being the epoch, -1 for continue, others hard indicating. For safety, if start epoch is the lastest as saved model, will quit ')
	parser.add_argument('--if_scraG', default='n', help='if backbone net from scratch')
	parser.add_argument('--init_type', default='xavier', help='weight initialization mode, gain 0.02 fixed in')
	parser.add_argument('--n_thread', default=10, type=int, help='how many threads')
	parser.add_argument('--save_step', default=2, type=int, help='how many steps to save model')
	parser.add_argument('--if_cmJoints', default='n', help='if use common joints across datasets without including the contriversal parts like sites and upper neck, torso, but only neck and head, pelvis will be always kept, obsolette')
	parser.add_argument('--if_pinMem', action='store_false', help='if pin memory to accelerate. Not working on windows')
	parser.add_argument('--if_finalTest', default='n', help='if run a final test and keep the result after training session')

	# -- visualization
	if if_discovery:
		parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
	else:
		parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
	parser.add_argument('--display_server', type=str, default="http://localhost",
	                    help='visdom server of the web display')
	parser.add_argument('--display_env', type=str, default='main',
	                    help='visdom display environment name (default is "main")')
	parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
	parser.add_argument('--use_html', action='store_true', help='if use html')
	parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
	parser.add_argument('--display_ncols', type=int, default=3,
	                    help='if positive, display all images in a single visdom web panel with certain number of images per row.')
	parser.add_argument('--update_html_freq', type=int, default=20,
	                    help='frequency of saving training results to html, def 1000 ')
	parser.add_argument('--print_freq', type=int, default=10,
	                    help='frequency of showing training results on console, def 100')
	parser.add_argument('--no_html', action='store_true',
	                    help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

	# -- test setting
	parser.add_argument('--testset', default='Human36M', help='testset, usually single [Human3dM|ScanAva]')
	parser.add_argument('--testIter', type=int, default=-1, help='test iterations final and epoch test, -1 for all, DBP')
	# parser.add_argument('--n_foldingLpTest', type=int, default=20, help='downsample the epoch test to quicken the in loop test process. 1 for full test in loop')
	parser.add_argument('--if_flipTest', default='y')
	parser.add_argument('--if_gtRtTest', default='y', help='if use gt distance for root')
	parser.add_argument('--if_adj', default='y', help='if adjust the root location to adapt different dataset')
	parser.add_argument('--if_aveBoneRec', default='y', help='if use average boneLento recover the estimation')
	parser.add_argument('--testImg', default=None, help='if indicate image, test will show the skeleton and 2d images of it')
	parser.add_argument('--rt_pelvisUp', default=0., type=float, help='move estimattion result pelvis up ratio, due to the joint definition difference. Apply at evaluation, main for h36m. scanava 0.08. ')
	parser.add_argument('--bone_type', default='h36m', help='choose the type for joint selection [scanava|h36m|cmJoints]')
	parser.add_argument('--if_loadPreds', default='n', help='if load preds in test func')
	parser.add_argument('--if_test_ckpt', default='n', help='if check intermediate checkpoint')
	parser.add_argument('--svVis_step', default=10, type=int, help='step to save visuals')
	parser.add_argument('--test_par', default='test', help='the exact test portion, could be [testInLoop|test|train]')  # I just save default first

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
	opts.ref_flip_pairs =ut_p.nameToIdx(opts.ref_flip_pairs_name, opts.ref_joints_name)
	opts.ref_root_idx = opts.ref_joints_name.index('Pelvis')
	opts.ref_evals_name = evals_name_config[opts.bone_type]
	if 'h36m' == opts.bone_type:
		opts.ref_skels_name = Human36M.skels_name
	else:
		opts.ref_skels_name = ScanAva.skels_name

	opts.ref_nEval = len(opts.ref_evals_name)
	opts.ref_skels_idx = ut_p.nameToIdx(opts.ref_skels_name, opts.ref_joints_name)
	opts.ref_evals_idx = ut_p.nameToIdx(opts.ref_evals_name, opts.ref_joints_name)

	opts.clipMode = '01'        # for save image purpose
	opts.adj_dict = {     # fill this if after calculation
		'Human36M': (0, 0., 0.),
		'MuPoTS': (0, 0, 0),
		'ScanAva': None,
		'SURREAL': None,
		'MSCOCO': None
	}
	opts.adj = opts.adj_dict[opts.testset]      # choose the one

	# Derived parameters, model, result part...
	# form exp folder
	if not os.path.isabs(opts.output_dir):
		opts.output_dir = os.path.abspath(opts.output_dir)
	opts.ref_joints_num = len(opts.ref_joints_name)  # how image output
	opts.ref_evals_num = len(opts.ref_evals_name)  # could be smaller
	nmT = '-'.join(opts.trainset)  # init
	suffix_train = (opts.suffix_ptn_train.format(
		**vars(opts))) if opts.suffix_ptn_train != '' else ''  # vars return __dict__ attribute
	nmT = '_'.join([nmT, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix
	opts.name = nmT     # current experiment name
	opts.exp_dir = osp.join(opts.output_dir, nmT)
	opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
	opts.vis_dir = osp.join(opts.exp_dir, 'vis', opts.test_par)
	opts.log_dir = osp.join(opts.exp_dir, 'log')
	opts.rst_dir = osp.join(opts.exp_dir, 'result')
	opts.num_gpus = len(opts.gpu_ids)
	opts.web_dir = osp.join(opts.exp_dir, 'web')
	opts.vis_test_dir = osp.join(opts.vis_dir, opts.testset)    # specific test dataset

	yn_dict = {'y': True, 'n': False}
	opts.flip_test = yn_dict[opts.if_flipTest]
	opts.use_gt_info = yn_dict[opts.if_gtRtTest]

	# for start epoch   name  [epoch]_net_[netName].pth
	# model_file_list = glob.glob(osp.join(opts.model_dir, '*.pth'))
	if osp.exists(opts.model_dir):
		model_file_list = [nm for nm in os.listdir(opts.model_dir) if nm.endswith('.pth')]
		if model_file_list:
			cur_epoch = max([ut_t.getNumInStr(fNm)[0] for fNm in model_file_list])
			start_epoch_sv = cur_epoch + 1
		else:
			start_epoch_sv = 0
		if opts.start_epoch == -1:
			opts.start_epoch = start_epoch_sv
		elif start_epoch_sv != opts.start_epoch and 'y' != opts.if_test_ckpt:
			print('not latest epoch, for protection concern, please clean up exp folder mannually ')
			exit(-1)
	else:       # no dir, first time
		opts.start_epoch=0      #

	# test name needs start_epoch
	sfx_test = (opts.suffix_ptn_test.format(**vars(opts))) if opts.suffix_ptn_test != '' else ''
	opts.nmTest = '_'.join((sfx_test, opts.suffix_exp_test))
	# otherwise, do nothing use the current start_epoch
	return opts


def print_options(opt, if_sv = False):
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
	if if_sv:
		ut_t.make_folder(opts.exp_dir)  # all option will mk dirs  # saved to json file in set_env
		file_name = os.path.join(opt.exp_dir, 'opts {}.txt'.format(opts.start_epoch))   # each train save one in case repurpose some model.
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
	if opts.cocoapi_dir:
		add_pypath(opts.cocoapi_dir)     # add coco dir to it
	add_pypath(osp.join('data', opts.testset))

	# add folders
	make_folder(opts.model_dir)
	make_folder(opts.vis_dir)
	make_folder(opts.log_dir)
	make_folder(opts.rst_dir)
	make_folder(opts.web_dir)

	# for continue, load previous setting to keep model consistency
	# if opts.start_epoch == 0:   # only first time save options, to stand for how model is trained on
	# 	if_sv = True
	# else:
	# 	if_sv = False
	print_options(opts, True)



