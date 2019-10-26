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



# opt.print_options(opts)
# print(opts.gpu_ids)

# get ds
# ds = Human36M('testInLoop')
# idx = 0

# skels_idx = ut_p.nameToIdx(Human36M.skels_name, Human36M.joints_name)
# print(skels_idx)
jt_idx = ut_p.nameToIdx(MPII.joints_eval_name, Human36M.joints_name)
print(jt_idx)


