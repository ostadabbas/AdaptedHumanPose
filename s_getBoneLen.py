from data.Human36M.Human36M import Human36M
from data.ScanAva.ScanAva import ScanAva
from data.SURREAL.SURREAL import SURREAL
from data.MuPoTS.MuPoTS import MuPoTS
import argparse
from opt import opts
import utils.utils_pose as ut_p


# ds = Human36M('test', opts)
# ds = ScanAva('test', opts)
ds = SURREAL('test', opts)
# print(type(ds).__name__ + 'average bone Len is')
# print(boneLen)

# new bone calculation
# ds = MuPoTS('test', opts)
# skel = ut_p.nameToIdx(opts.ref_skels_name, ds.joints_name)
# boneLen_dict = ut_p.get_boneLen_av(ds.data, skel=skel, fn_getIdx=ds.getNmIdx)
# print(boneLen_dict)

