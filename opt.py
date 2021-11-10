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
	parser.add_argument('--ds_dir', default='/scratch/liu.shu/datasets', help='dataset directionry')      # for discovery
	parser.add_argument('--output_dir', default='output', help='default output dirs') # code local.
	parser.add_argument('--cocoapi_dir', default=None, help='coco api dir. If cocoapi is not installed as a package, you can point this directly to that folder')
	parser.add_argument('--ifb_debug', action='store_true')
	parser.add_argument('--suffix_ptn_train', default='{net_BB}_D{lmd_D}-l{n_layers_D}k{kn_D}-{mode_D}-p{pivot}_{gan_mode}_yl-{if_ylB}_regZ{epoch_regZ}_fG-{if_fixG}_lr{lr}', help='the suffix pattern to form name') ## -- no p{if_pivlt} before
	parser.add_argument('--suffix_exp_train', default='exp', help='the manually given suffix for specific test')
	parser.add_argument('--suffix_ptn_test', default='{testset}_Btype-{bone_type}_SBL-{smplBL}_PA-n', help='the suffix pattern to form name') # add simple baselline, start will change after training only one
	parser.add_argument('--suffix_exp_test', default='exp', help='mannualy added suffix for test result')
	parser.add_argument('--model', default='TaskGenNet', help='mannualy added suffix for test result')  # for MPPI:  MPPE_Human36M_p1/p2 or  MPPE_MuCo  depend what is trained on, hg3d  for the  MPI version
	# -- ds specific
	parser.add_argument('--h36mProto', default=2, type=int)

	# -- train setting
	parser.add_argument('--trainset', nargs='+', default=['ScanAva', 'MSCOCO', 'MPII'], help='give the main ds here the iter number will follow this one')
	parser.add_argument('--gan_mode', default='lsgan', help='gan type [lsgan|vanilla]')
	parser.add_argument('--lmd_D', default=0.02, type=float, help='weight for D loss, 0. for no D')
	parser.add_argument('--mode_D', default='SA', help='the discriminator type, [SA|C|C1], semantic aware or conventional, whole patch, C1: image 1 label ')
	# parser.add_argument('--stgs_D', nargs='+', default=[1, 0, 0, 0], type=int, help='indicator to show which stage is requested for assimilation,  last 4 layers in rest net ') # no used  no multi-stg gau
	# parser.add_argument('--gauss_sigma', type=float, default=2.0,help='gaussian sigma to draw the heat map')
	# parser.add_argument('--if_scale_gs', default='n', help='if up scale gaussian for higher level  features map, otherwise direct downsample. So higheest hm still hold small values')        # not using now
	parser.add_argument('--if_ylB', default='y', help='if train the low task in target B domain')
	parser.add_argument('--pivot', default='sdt', help='pivoting mode, n: no, only on real.  uda: reverse label, sdt: even dist ')
	parser.add_argument('--if_tightBB_ScanAva', default='y', help='if use tight bb for scanAva')
	parser.add_argument('--rt_SYN', default=1, type=int, help='the ratio of the subjects used in training, larger step will make less training data')
	parser.add_argument('--inp_sz', default=256, type=int, help='input image size')
	parser.add_argument('--end_epoch', default=15, type=int, help='when reach this epoch, will stop. python index style, your model will be saved as epoch_tar-1, ori 25, 15 for quick then 25.')
	parser.add_argument('--epoch_step', default=-1, type=int, help='mainly for time constrained system, each time only train certain epoch steps, -1 for all')
	parser.add_argument('--trainIter', default=2500, type=int, help='train iters each epoch, -1 for whole set. For debug purpose (DBP)')    # 82000 /120 = 2050  for ScanAva
	parser.add_argument('--if_normBone', default='n', help='true of false [y|n] normalized the bones')  # not  used  ---
	parser.add_argument('--if_fixG', default='n', help='if fix G after introducing z training')
	parser.add_argument('--epoch_regZ', default=5,  type=int, help='when to start training z part. depend on option to see if fixG or not. 0 means from very beginning')
	parser.add_argument('--optimizer', default='adam', help='[adam|nadam]')
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--lr_policy', default='multi_step', help='[step|plateau|multi_step|cosine]')
	parser.add_argument('--lr_dec_epoch', nargs='+', type=int, default=[11, 13], help='the lr decay epoch, each time by decay factor ori 17,21')   # form string sec dec17-21 if needed, 11, 13?
	parser.add_argument('--batch_size', default=10, type=int)       # 120 to 20 for saving more images.
	# test batch size 16 what is the reason,, no idea
	parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='the ids of the gpu')
	parser.add_argument('--start_epoch', default=-1, type=int, help='where to being the epoch, -1 for continue, others hard indicating. For safety, if start epoch is the lastest as saved model, will quit ')
	parser.add_argument('--if_scraG', default='n', help='if backbone net from scratch')
	parser.add_argument('--init_type', default='xavier', help='weight initialization mode, gain 0.02 fixed in')
	parser.add_argument('--n_thread', default=10, type=int, help='how many threads')
	parser.add_argument('--save_step', default=3, type=int, help='how many steps to save model')
	parser.add_argument('--if_cmJoints', default='n', help='if use common joints across datasets without including the contriversal parts like sites and upper neck, torso, but only neck and head, pelvis will be always kept, obsolette') #
	parser.add_argument('--if_pinMem', action='store_false', help='if pin memory to accelerate. Not working on windows')
	parser.add_argument('--if_finalTest', default='y', help='if run a final test and keep the result after training session')

	# -- visualization
	parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
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
	parser.add_argument('--testset', default='Human36M', help='testset, usually single [Human3dM|ScanAva|MuPoTS|SURREAL]')
	parser.add_argument('--testIter', type=int, default=-1, help='test iterations final and epoch test, -1 for all, DBP')
	parser.add_argument('--if_flipTest', default='y')
	parser.add_argument('--if_gtRtTest', default='y', help='if use gt distance for root')
	parser.add_argument('--if_aveBoneRec', default='y', help='if use average boneLento (from alll dataset) recover the estimation. Better as the 2d can have some errors')
	# only for norm bone case
	parser.add_argument('--rt_pelvisUp', default=0., type=float, help='move estimattion result pelvis up ratio, due to the joint definition difference. Apply at evaluation, main for h36m. scanava 0.08. ')  # not used  --
	parser.add_argument('--bone_type', default='h36m', help='choose the type for joint selection [scanava|h36m|cmJoints], what to be trained')
	parser.add_argument('--if_loadPreds', default='n', help='if load preds result in test func, already saved data to accelerate')
	parser.add_argument('--if_test_ckpt', default='n', help='if check intermediate checkpoint not the last one') # not used too much,  usually on check final verison..
	parser.add_argument('--svVis_step', default=5, type=int, help='step to save visuals')
	parser.add_argument('--test_par', default='test', help='the exact test portion, could be [testInLoop|test|train], can use the model to test on train set or test set')  # I just save default first
	parser.add_argument('--smplBL', type=str, default='n', help='if given, the depth is estimated with the simplBL model from specific ds, [ds_cam|ds|n]')  # I just save default firs
	# parser.add_argument('--if_PA', type=str, default='n', help='if use pose adaptation')  # I just save default first, all in hm space, as you are not always have the camera parameters, but recover in a hm space.

	# -- network settings
	parser.add_argument('--n_layers_D', type=int, default=3, help='descriminator layer number, for 8 bb, 2 layers are good')
	parser.add_argument('--kn_D', type=int, default=1, help='kernel size for pixel one ')
	parser.add_argument('--net_BB', default='res50', help='backbone net type [res50|res101|res152], can extended to VGG different layers, not so important , add later')

	# -- PA net
	parser.add_argument('--end_epoch_PA', type=int, default=70, help='the end epoch of PA net') # --
	parser.add_argument('--batch_size_PA', type=int, default=256 , help='the end epoch of PA net, default 64 ')  # --
	parser.add_argument('--tarset_PA', default='h36m-p2', help='the target dataset for the PA net [h36p1, h36p2, MuCo], will load the target sets') # *** model for MPPE
	parser.add_argument('--if_gt_PA', default='n', help='if use the tarset gt skel to guide the PA') #  most tims similar
	parser.add_argument('--if_continue_PA', default='n', help='if continue from scrath of PA, or try to load ') # --
	parser.add_argument('--if_VPR', default='y', help='if use viewpoint randomization') #  only scale
	parser.add_argument('--if_test_PA', default='n', help='if only test the PA net, no training will happen')    # --
	parser.add_argument('--if_norm_PA', default='n', help='if normalize the PA vector.n') # -- keep original unit is better, normalize no much help
	parser.add_argument('--if_neckRt', default='n', help='if neck rooted during mapping.') # -- keep original unit is better
	parser.add_argument('--if_hm_PA', default='y', help='if run the pose adaptation at hm level.')  # -- always more practical
	parser.add_argument('--decay_PA', nargs='+',  type=int, default=[160, 180], help='the decay step') # --
	## with GD study
	parser.add_argument('--PA_G_mode',  type=int, default=2, help='the PA G type, in G/D study') # 1 for dirc map,  2 for res map
	parser.add_argument('--lr_PA',  type=float, default=1e-4, help='the PA G type, in G/D study')
	parser.add_argument('--gamma_PA',  type=float, default=0.95, help='gamma for reducing the learning rate')
	parser.add_argument('--lmd_G_PA',  type=float, default=1., help='GD lambda') #
	parser.add_argument('--lmd_D_PA',  type=float, default=0., help='the PA G type, in G/D study')
	parser.add_argument('--lmd_skel_PA',  type=float, default=50., help='the skels punisher') #
	parser.add_argument('--if_gtSrc',  default='y', help='if use ground truth source training data, otherwise the pred result')
	parser.add_argument('--if_skel_av',  default='y', help='if using average skeleton') # y , more practical, from measure
	parser.add_argument('--if_skel_MSE',  default='y', help='if using MSE loss for skel loss instead of L1 ')  # use MSE
	parser.add_argument('--if_L_MSE',  default='n', help='for the pose regulator, if MSE or L1') #  use L1
	parser.add_argument('--if_clip_grad',  default='y', help='if clip the grad of G parameters') # 1 for dirc map,  2 for res


	# hardwired parameters
	# opts = parser.parse_args()   # all cmd infor
	opts, _ = parser.parse_known_args()   # all cmd infor  # only known, 2nd part for f
	# print(opts)
	# print(opts.inp_sz)

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

	# auo the opts
	opts.input_shape = (opts.inp_sz, opts.inp_sz)  # tuple size
	opts.output_shape = [opts.input_shape[0] // 4, opts.input_shape[1] // 4]
	opts.depth_dim = opts.input_shape[0] // 4  # save as output shape, df 64.
	opts.bbox_3d_shape = (2000, 2000, 2000)  # depth, height, width
	opts.pixel_mean = (0.485, 0.456, 0.406)  # perhaps for RGB normalization  after divide by 255
	opts.pixel_std = (0.229, 0.224, 0.225)

	opts.ref_joints_name = Human36M.joints_name  # stick to Human36M, we can not evaluate but keep all, ref for train
	opts.ref_flip_pairs_name = Human36M.flip_pairs_name
	opts.ref_flip_pairs = ut_p.nameToIdx(opts.ref_flip_pairs_name, opts.ref_joints_name)
	opts.ref_root_idx = opts.ref_joints_name.index('Pelvis')
	opts.ref_evals_name = evals_name_config[opts.bone_type]  # which one for evaluation. for final test.
	if 'h36m' == opts.bone_type:
		opts.ref_skels_name = Human36M.skels_name       # 17
	else:
		opts.ref_skels_name = ScanAva.skels_name        # fewer jts

	opts.ref_nEval = len(opts.ref_evals_name)
	opts.ref_skels_idx = ut_p.nameToIdx(opts.ref_skels_name, opts.ref_joints_name)
	opts.ref_evals_idx = ut_p.nameToIdx(opts.ref_evals_name, opts.ref_joints_name)

	opts.clipMode = '01'  # for save image purpose
	opts.adj_dict = {  # fill this if after calculation, not used this time
		'Human36M': (0, 0., 0.),
		'MuPoTS': (0, 0, 0),
		'MuCo': (0, 0, 0),
		'ScanAva': None,
		'SURREAL': None,
		'MSCOCO': None
	}
	opts.adj = opts.adj_dict[opts.testset]  # choose the one

	# Derived parameters, model, result part...
	# form exp folder
	if not os.path.isabs(opts.output_dir):
		opts.output_dir = os.path.abspath(opts.output_dir)
	opts.ref_joints_num = len(opts.ref_joints_name)  # how image output
	opts.ref_evals_num = len(opts.ref_evals_name)  # could be smaller
	nmT = '-'.join(opts.trainset)  # init
	dct_opt = vars(opts)
	# set tne naming needed attirbutes
	suffix_train = (opts.suffix_ptn_train.format(
		**vars(opts))) if opts.suffix_ptn_train != '' else ''  # vars return __dict__ attribute
	nmT = '_'.join([nmT, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix
	if 'MPPE' in opts.model or 'hg3d' in opts.model or 'evoSkel' in opts.model:        # specific for  the MPPE test session
		nmT = opts.model    # directly maped to it
	else:
		opts.name = nmT  # current experiment name
	opts.exp_dir = osp.join(opts.output_dir, nmT)
	opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
	opts.vis_dir = osp.join(opts.exp_dir, 'vis', opts.test_par)  # exp/vis/partition/Human36M/G_fts_raw
	opts.log_dir = osp.join(opts.exp_dir, 'log')
	opts.rst_dir = osp.join(opts.exp_dir, 'result')
	opts.num_gpus = len(opts.gpu_ids)
	opts.web_dir = osp.join(opts.exp_dir, 'web')
	opts.vis_test_dir = osp.join(opts.vis_dir, opts.testset)  # specific test dataset

	yn_dict = {'y': True, 'n': False}
	opts.flip_test = yn_dict[opts.if_flipTest]
	opts.use_gt_info = yn_dict[opts.if_gtRtTest]

	if osp.exists(opts.model_dir):
		model_file_list = [nm for nm in os.listdir(opts.model_dir) if
		                   nm.endswith('.pth')]  # can load in check the epoch key
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
	else:  # no dir, first time
		opts.start_epoch = 0  #

	# test name needs start_epoch
	sfx_test = (
		opts.suffix_ptn_test.format(**vars(opts))) if opts.suffix_ptn_test != '' else ''  # bonetype_dsTest_epoch_split
	# if opts.if_VPR == 'y' and opts.if_PA == 'y':  # only use pa and vpr add vpr to it.
	# 	sfx_test += '_VPR'
	opts.nmTest = '_'.join((sfx_test, opts.suffix_exp_test))

	# add folders
	make_folder(opts.model_dir)
	make_folder(opts.vis_dir)
	make_folder(opts.log_dir)
	make_folder(opts.rst_dir)
	make_folder(opts.web_dir)

	print_options(opts, True)



