'''
plot loss out. 1st version used for discovery
'''
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
import re
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt

output_dir = '../output'       # DISC version
mdl_ptn = 'ScanAva-MSCOCO-MPII_res50_n-scraG_{}D2_y-yl_1rtSYN_regZ0_n-fG_n-nmBone_adam_lr0.001_exp'
pth_ptn = osp.join(output_dir, mdl_ptn, 'log', 'train_logs.txt')
d_li = ['0.0', '0.01', '1.0', '10.0','100.0', '1000.0']
# rst_fd = r'S:\ACLab\rst_model\taskGen3d\output\ScanAva-MSCOCO-MPII_res50_n-scraG_0.1D2_y-yl_1rtSYN_regZ10_n-fG_y-nmBone_adam_lr0.001_exp'
# trainLog_pth = osp.join(rst_fd, 'log', 'train_logs.txt')
smpl_rt = 100
# show loss
	# get the loss list
for d in d_li:
	pth = pth_ptn.format(d)
	loss_li = []
	for i, line in enumerate(open(pth)):
		if i % smpl_rt != 0:
			continue
		m = re.search('loss:T:(\d\.\d+)', line, re.IGNORECASE)
		if m:
			loss_li.append(float(m.group(1)))
	plt.plot(medfilt(loss_li, 21))

plt.legend(d_li)
plt.show()
