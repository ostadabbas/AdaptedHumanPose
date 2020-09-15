'''
for the pkg function test
'''

from scipy.spatial.transform import Rotation as R
import random
import numpy as np

## test rotation and transformation.
# x_r = 30
# y_r = 180
# z_r = 30
arr1 = np.arange(15).reshape([5,3])
rt = R.from_euler('xyz', angles=[30,0,0], degrees=True)
print(rt.as_matrix())       # 1.5 installed
# rst = rt.apply(arr1)        # numpy nd array apply to each row
# print(rst,'type is', type(rst))

## numpy random
