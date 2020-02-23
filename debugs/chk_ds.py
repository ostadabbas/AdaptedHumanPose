'''
check the dataet wholesome
'''
# add proj path
import sys, os
proj_pth = os.path.join(os.path.dirname(sys.path[0]))
if proj_pth not in sys.path:
	sys.path.insert(0, proj_pth)
from opt import opts
from data.SURREAL.SURREAL import SURREAL
from data.MuCo.MuCo import MuCo
from data.ScanAva.ScanAva import ScanAva
from data.dataset import AdpDataset_3d
from tqdm import tqdm
import os.path as osp
import utils.utils_tool as ut_t
import numpy as np
import cv2

if __name__=='__main__':
	if_train= True
	# ds = SURREAL('train', opts)
	ds = MuCo('train', opts)
	# ds = ScanAva('train', opts)
	idx_st = 0
	ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, if_train, opts=opts)

	N = ds_adp.__len__()
	tqdm_bar = tqdm(total=ds_adp.__len__())

	## random plot
	h = 2048
	n_smpl = 50
	# cycl_size = int(h/100.)
	# scal_test = h * 0.0015
	cycl_size = 5
	scal_test = 0.5
	offset = 0

	testDir = osp.join('test', type(ds).__name__ +'_smpls')
	ut_t.make_folder(testDir)

	smpls = np.random.choice(ds.data, n_smpl)
	for i in tqdm(range(n_smpl)):
		smpl = smpls[i]
		img_pth = smpl['img_path']
		img = cv2.imread(img_pth)
		joint_img = smpl['joint_img'].astype(int)
		bsNm = osp.basename(img_pth)
		for j, joint in enumerate(joint_img):
			cv2.circle(img, (joint[0], joint[1]), cycl_size, (0, 255, 0), -1)
			# print(joint)
			# print(offset, type(offset))
			cv2.putText(img, str(j), (joint[0] + offset, joint[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, scal_test, (255, 255, 255))
		cv2.imwrite(osp.join(testDir, 'sample_{}.png'.format(i)), img)


	##  check SURREAL stuck issue,   loop all data
	# jt_shp_pre = (17,3)
	# vis_shp_pre = (17,)
	# depth_shp_pre = (17,)
	# SYN_shp_pre = (17,)
	# img_shp_pre = None
	# for i, (inp, tar) in tqdm(enumerate(ds_adp)):        # loop ds directly
	# for i in tqdm(range(idx_st, N)):      # only check the needed N
	# 	inp, tar = ds_adp.__getitem__(i)
	# 	img_patch = inp['img_patch']
	# 	joint_hm = tar['joint_hm']
	# 	vis_v = tar['vis']
	# 	if_depth_v = tar['if_depth_v']
	# 	if_SYN_v = tar['if_SYN_v']
	# 	img_shp_cur = img_patch.size()
	# 	jt_shp_cur = joint_hm.shape
	# 	vis_shp_cur = vis_v.shape
	# 	depth_shp_cur = if_depth_v.shape
	# 	SYN_shp_cur = if_SYN_v.shape
	# 	if img_shp_pre:  # check previous one and current size
	# 		if img_shp_cur != img_shp_pre or jt_shp_cur != jt_shp_pre or vis_shp_cur!=vis_shp_pre or depth_shp_cur!= depth_shp_pre or SYN_shp_cur!=SYN_shp_pre:
	# 			print('shape differes at  {}'.format(i))
	# 			print('image shape {}'.format(img_shp_cur))
	# 			print('jt shape {}'.format(jt_shp_cur))
	# 			print('vis shape {}'.format(vis_shp_cur))
	# 			print('depth shape {}'.format(depth_shp_cur))
	# 			print('SYN_shp_cur shape {}'.format(SYN_shp_cur))
	# 	img_shp_pre = img_shp_cur
	# 	jt_shp_pre = jt_shp_cur
	# 	vis_shp_pre = vis_shp_cur
	# 	depth_shp_pre = depth_shp_cur
	# 	SYN_shp_pre = SYN_shp_cur
	# 	# tqdm_bar.update()
