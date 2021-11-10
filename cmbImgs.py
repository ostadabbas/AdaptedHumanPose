'''
combine 2d and 3d images
'''
import sys, os
# proj_pth = os.path.join(os.path.dirname(sys.path[0]))
# if proj_pth not in sys.path:
# 	sys.path.insert(0, proj_pth)

import cv2
import utils.utils_tool as ut_t
import os.path as osp
import numpy as np
from tqdm import tqdm
from opt import opts

if __name__ == '__main__':
	vis_dir = opts.vis_dir
	ts_sets = os.listdir(vis_dir)
	tar_size = (256, 256)
	for set in ts_sets:
		set_fd = osp.join(vis_dir, set)
		cmb_fd = osp.join(set_fd, '2d3d')
		ut_t.make_folder(cmb_fd)
		fd_2d = osp.join(vis_dir, set, '2d')
		f_nms = os.listdir(fd_2d)
		for nm in tqdm(f_nms, desc='combining {}'.format(set)):
			img_pth = osp.join(set_fd, '2d', nm)
			img_2d = cv2.imread(img_pth)
			img_pth = osp.join(set_fd, '3d_hm', nm)
			img_3d = cv2.imread(img_pth)
			img_2d = cv2.resize(img_2d, tar_size)
			img_3d = cv2.resize(img_3d, tar_size)

			img_cmb = np.concatenate([img_2d, img_3d], axis=1)
			cv2.imwrite(osp.join(cmb_fd, nm), img_cmb)

