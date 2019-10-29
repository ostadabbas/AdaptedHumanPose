'''
project module test purpose
'''
import opt
import utils.utils_tool as ut_t
opts = opt.opts
ut_t.add_pypath(opts.cocoapi_dir)       # add coco path
from data.Human36M.Human36M import Human36M
from data.MPII.MPII import MPII
from utils import vis
import utils.utils_pose as ut_p
import utils.utils_tool as ut_t
import numpy as np
from skimage import io, transform
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from utils.logger import Colorlogger


# opt.print_options(opts)
# print(opts.gpu_ids)

# get ds
# ds = Human36M('testInLoop')
# idx = 0

# skels_idx = ut_p.nameToIdx(Human36M.skels_name, Human36M.joints_name)
# print(skels_idx)
# jt_idx = ut_p.nameToIdx(MPII.joints_eval_name, Human36M.joints_name)
# print(jt_idx)

# test the resize and grid
# img1_c6 = np.ones((6, 3,3))
# scale_vec = np.arange(6)
# img1_c6 = img1_c6 * scale_vec.reshape((6,1, 1))
# fts_rsz = transform.resize(img1_c6.transpose((1, 2, 0)), (60, 60)).transpose((2, 0, 1))
# print(fts_rsz.shape)
# grid = ut_t.gallery(fts_rsz, ncols=3)
# plt.imshow(grid)
# plt.show()        # -- ok

## test logger
arr1 = np.array([[2,8,6],
                 [9,8, 1]])
logger_test = Colorlogger(opts.log_dir, 'test_logs.txt')
logger_test.info(arr1)






