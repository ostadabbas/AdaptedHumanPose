'''
collect result from tested model.
taskGen3d saved names,  rst: (raw pred),
eval: p1_err, p2_err, z1,z2,
3dpck:  pck_v_tot, auc_tot
 default='{testset}_Btype-{bone_type}_SBL-{smplBL}'
 name for h36m:
  ds_test.data_split, 'flip_{}'.format(opts.if_flipTest),'proto' + str(ds_test.protocol), 'pred_hm.npy']))  #
  first round save tab version for detail.
  Then save the lax version from selected
'''
import os.path as osp
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--synset', default='ScanAva', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--gan_mode', default='lsgan', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--mode_D', default='SA', help='the gan mode [lsgan|vanilla]')
parser.add_argument('--testset', default='Human36M', help='the gan mode [Human36M|MuPoTS]')
parser.add_argument('--proto', type=int, default=2, help='the gan mode [lsgan|vanilla]')    # proto 2
parser.add_argument('--SBL', default='n', help='if use simple baseline.')
parser.add_argument('--lmd_D', type=float, default=0.02, help='if use simple baseline.')
opt = parser.parse_args()

testset = opt.testset
proto = opt.proto
lmd_D = opt.lmd_D
gan_mode = opt.gan_mode
mode_D = opt.mode_D
synset = opt.synset
SBL = opt.SBL
# Human36M_Btype-h36m_SBL-n_exp_test_proto2, give option to take into consideration of SBL and PA.
if 'Human36M' == testset:
	pthHd = 'Human36M_Btype-h36m_SBL-{}_PA-n_exp_test_proto{}'.format(SBL, proto)      # part kept
	outNm = 'Human36M_p{}'.format(proto)
else:
	pthHd = '{}_Btype-h36m_SBL-{}_PA-n_exp_test'.format(SBL, testset)
	outNm = '{}'.format(testset)

eval_nm = pthHd + '_eval.json'
pck_nm = pthHd + '_3dpck.json'

outFd = 'output/metricRst'
if not osp.exists(outFd):
	os.makedirs(outFd)
# no p old one
# exp_li = [
# 	'ScanAva-MSCOCO-MPII_res50_D0.0-SA-lsgan_yl-n_regZ5_fG-n_lr0.001_exp',      # raw
# 	'ScanAva-MSCOCO-MPII_res50_D{}-C-lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),      # C
# 	'ScanAva-MSCOCO-MPII_res50_D{}-SA-lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),     # SA
# 	'ScanAva-MSCOCO-MPII_res50_D{}-SA-lsgan_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D),     # SA-yB
# 	'ScanAva-MSCOCO-MPII_res50_D0.0-SA-lsgan_yl-y_regZ5_fG-n_lr0.001_exp',     # D0 joint
#
# 	'SURREAL-MSCOCO-MPII_res50_D0.0-SA-lsgan_yl-n_regZ5_fG-n_lr0.001_exp',      # raw
# 	'SURREAL-MSCOCO-MPII_res50_D{}-C-lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),      # C
# 	'SURREAL-MSCOCO-MPII_res50_D{}-SA-lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),     # SA
# 	'SURREAL-MSCOCO-MPII_res50_D{}-SA-lsgan_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D),     # SA-yB
# 	'SURREAL-MSCOCO-MPII_res50_D0.0-SA-lsgan_yl-y_regZ5_fG-n_lr0.001_exp',     # D0 joint
# ]


# real one
# exp_li = [
# 	"Human36M-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-y_regZ5_fG-n_lr0.001_exp",
# 	"MuCo-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-y_regZ5_fG-n_lr0.001_exp",
# ]

# joint training yB-y
# exp_li = [
# 	# "ScanAva-MSCOCO-MPII_res50_D0.0-SA-py_lsgan_yl-n_regZ5_fG-n_lr0.001_exp",     # original in 1.0
# 	# "ScanAva-MSCOCO-MPII_res50_D0.0-SA-py_lsgan_yl-y_regZ5_fG-n_lr0.001_exp",
# 	"ScanAva-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-n_regZ5_fG-n_lr0.001_exp",
# 	"ScanAva-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-y_regZ5_fG-n_lr0.001_exp",
# ]

# test p-n, uda, sdt, l2k3
# exp_li = [
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-C-pn_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-C-puda_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-C-psdt_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-C-psdt_lsgan_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-SA-pn_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-SA-puda_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-SA-psdt_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# 	'ScanAva-MSCOCO-MPII_res50_D{}-l2k3-SA-psdt_lsgan_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D),
# ]

# test p-n, l3k1
exp_li = [
	# '{}-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset),        # fixed  no D version
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-C-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	# '{}-MSCOCO-MPII_res50_D{}-l3k1-SA-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
	'{}-MSCOCO-MPII_res50_D{}-l3k1-SA-pn_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(synset, lmd_D, gan_mode),
]

# C1, SA1 test
# exp_li = [
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-{}-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, mode_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-{}-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, mode_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-{}-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, mode_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-{}-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D, mode_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-SA1-pn_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-SA1-puda_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-SA1-psdt_{}_yl-n_regZ5_fG-n_lr0.001_exp'.format(lmd_D, gan_mode),
# 	# 'ScanAva-MSCOCO-MPII_res50_D{}-l3k1-SA1-psdt_{}_yl-y_regZ5_fG-n_lr0.001_exp'.format(lmd_D, gan_mode),
# ]
out_pth = os.path.join(outFd, outNm+'.txt')
mp_ltx_pth = os.path.join(outFd, outNm + '_mpjpe.txt')      # 3 path for latext format
mp_pa_ltx_pth = os.path.join(outFd, outNm + '_mpjpe-pa.txt')
pck_ltx_pth = os.path.join(outFd, outNm + '_3dpck.txt')

print('save rst to {}'.format(out_pth))
f = open(out_pth, "w+")  # create new

#  Human36M_Btype-h36m_SBL-n_exp_test_proto2
# saving snippet
# err_dict = {
# 	'p1_err': p1_err_av,
# 	'p2_err': p2_err_av,
# 	'z1_err': z1_err_av,
# 	'z2_err': z2_err_av,
# 	'ref_evals_names': opts.ref_evals_name,  # what joints evaluated
# }
# # pck_rst = {'pck_v_tot':pck_v_tot, 'pck_v_li':pck_v_li, 'auc_tot':auc_tot, 'auc_li':auc_li}
# if if_align:  # pck only exists when aligned, otherwise not exists
# 	pck_rst = {'pck_v_tot': pck_v_tot, 'auc_tot': auc_tot}
li_eval = ['p1_err', 'p2_err', 'z1_err', 'z2_err']
li_pck = ['pck_v_tot', 'auc_tot']

for exp in exp_li:
	rstPth = osp.join('output', exp, 'result', eval_nm)
	with open(rstPth) as f_in:
		eval = json.load(f_in)

	rstPth = osp.join('output', exp, 'result', pck_nm)
	with open(rstPth) as f_in:
		pck3d = json.load(f_in)
		pck3d['pck_v_tot'] = pck3d['pck_v_tot'][-1]     # the 10cm pck, 15 cm
	for metric in li_eval:
		f.write('{}\t'.format(eval[metric]))
	for metric in li_pck:
		if metric == 'auc_tot': # if last
			f.write('{}'.format(pck3d[metric]))
		else:
			f.write('{}\t'.format(pck3d[metric]))
	f.write('\n')
f.close()

# open loop exp, get eval, write line  \n
with open(mp_ltx_pth, 'w') as f:
	for exp in exp_li:
		rstPth = osp.join('output', exp, 'result', eval_nm)
		with open(rstPth) as f_in:
			eval = json.load(f_in)
		f_in.close()
		p_err = eval['p2_err_act']    # for mpjpe
		p_av = eval['p2_err']
		msg = ' '.join(['{:4.1f} &'.format(err) for err in p_err])
		msg += '{:4.1f} \n'.format(p_av)
		f.write(msg)
	f.close()
print('save to {}'.format(mp_ltx_pth))

# open loop exp, get eval, write line  \n
with open(mp_pa_ltx_pth, 'w') as f:
	for exp in exp_li:
		rstPth = osp.join('output', exp, 'result', eval_nm)
		with open(rstPth) as f_in:
			eval = json.load(f_in)
		f_in.close()
		p_err = eval['p1_err_act']  # for mpjpe
		p_av = eval['p1_err']
		msg = ' '.join(['{:4.1f} &'.format(err) for err in p_err])
		msg += '{:4.1f} \n'.format(p_av)
		f.write(msg)
	f.close()
print('save to {}'.format(mp_pa_ltx_pth))

with open(pck_ltx_pth, 'w') as f:
	for exp in exp_li:
		rstPth = osp.join('output', exp, 'result', pck_nm)
		with open(rstPth) as f_in:
			eval = json.load(f_in)
		f_in.close()
		p_err = eval['pck_v_li']  # for mpjpe
		p_err = [e[-1] for e in p_err]  # only take last
		p_av = eval['pck_v_tot'][-1]
		auc_av = eval['auc_tot']
		msg = ' '.join(['{:4.1f} &'.format(err*100) for err in p_err])
		msg += '{:4.1f} &'.format(p_av*100)
		msg += '{:4.1f} \n'.format(auc_av*100)
		f.write(msg)
	f.close()
print('save to {}'.format(pck_ltx_pth))