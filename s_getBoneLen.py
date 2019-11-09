from data.Human36M.Human36M import Human36M
from data.ScanAva.ScanAva import ScanAva
from data.SURREAL.SURREAL import SURREAL
import argparse
from opt import opts

# ds = Human36M('test', opts)
# ds = ScanAva('test', opts)
ds = SURREAL('test', opts)
boneLen = ds.getBoneLen_av()
print(type(ds).__name__ + 'average bone Len is')
print(boneLen)

