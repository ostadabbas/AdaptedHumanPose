'''
check the reuslt to compare
'''
import numpy as np
import json
import os.path as osp

## check the hm result, wh large av_err but nearly identical to original  ?
# idx = 0

# rst_dir = 'output/ScanAva-MSCOCO-MPII_res50_D0.02-l3k1-SA-psdt_lsgan_yl-y_regZ5_fG-n_lr0.001_exp/result/'
# PA_nm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto2_pred_hm_PA2.npy'
# gt_nm = 'Human36M_Btype-h36m_SBL-n_PA-n_exp_test_proto2_pred_hm.json'
# preds_PA = np.load(osp.join(rst_dir, PA_nm))
# with open(osp.join(rst_dir, gt_nm), 'r') as f:
# 	dtIn = json.load(f)
# 	f.close()
# preds = np.array(dtIn['pred'])  # pred gt of the h36m trainset
# gts = np.array(dtIn['gt'])
# print('preds_PA {}'.format(idx), preds_PA[idx])
# print('preds {}'.format(idx), preds[idx])
# print('gts  {}'.format(idx), gts[idx])
# print('diffs', preds_PA - preds)

pth = 'tmp/err_data.npz'
dt_in = np.load(pth)


