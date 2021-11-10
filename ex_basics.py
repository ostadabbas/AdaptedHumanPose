'''
test the basic comman,
'''
import numpy as np
import torch
import cv2
from time import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import os.path as osp
import json
# import ex_check     # there is np in ex_check, but can't work here
# from ex_check import np     # can import the module imported in another file



## basic
# if 0:
# 	print('0 is true')
# else:
# 	print('0 is not true')      # 0 not ture  check
# the operator &
# rst_andS = 1>0 & 2<3 & 2<4
# rst_andS = 4>0 & 100<8      # & first, then later inequality first   from back to front
# print(rst_andS) # true
# ls = [ 1, 2, 5]
# ls[1:] =3       # can only assign one iterable
# print(ls)
# li1 = list(range(8))
# li2  = li1[2,3,6]       # integer or slice can't individual
# print(li2)

## numpy
# arr1 = np.arange(12).reshape(2,2,3)
# print(arr1)
# arr2 = arr1[:,0]
# arr3 = arr1[:, (0,)]
# print('arr2 is', arr2)      # 2d 0 1 2 6 7 8
# print('arr3 is', arr3)      # 3d 2 x 1  x3
# print('minus all c first row', arr1-arr3)

## string
# print('{} haha'.format([1,2,3])) # formated, [1, 2, 3] haha
# pth = 'output/SURREAL-MSCOCO-MPII_res50_D0.02-l3k1-C-pn_lsgan_yl-n_regZ5_fG-n_lr0.001_exp/vis/test/Human36M/G_fts_raw/3000.npy'
# rst_toInt = int(pth)      # can't change directly
# li_int = [int(wd) for wd in pth.split('/') if wd.isdigit()]      # empty all connected, use '/' seperation, empty too
# li_int = [int(wd) for wd in pth.split('/')]      # must has space
# rst = int(pth.split('/')[-1][:-4])
# print(type(rst))

## list
# li1= []
# li2 = [torch.nn.Linear(2,3)]
# print(li1 + li2)
# print(type(li2) is list)
# li = ['ha', 'ba', 'gou']
# print(li)
# print(*li)
# li1 = list(range(10))
# print(li1[::3])     # can be sliced with stride (step)

## np test
# arr1 = np.ones([6, 2])      # get the first length first
# arr1 = np.array([2,3])      # get the first length first
# print(arr1**2)
# for i, e in enumerate(arr1):    # can be looped like this
# 	print('{}th number is {}'.format(i, e))
# arr1 = np.arange(18)
# arr2 = np.reshape(arr1, (2,3,3))
# arr3 = arr2.reshape((2,-1))
# mu = arr2.mean(axis=0)
# std = arr2.std(axis=0)
# nm1 = (arr2-mu)/std
# print(arr2[:,:2].flatten())     # partial flatten worked
# print(arr1)
# print(arr2)
# print(arr3)
# print('mu', mu)
# print('std', std)
# print(nm1)      # normalized
# print(arr2.copy())
# arr1 = np.zeros([2]+(3,4))  # can't combine turple and list
# arr1 = np.zeros([2]+list([3,4]))  # can't combine turple and list
# r1 = arr1[1]
# r1 = r1 + 1     # will not affect original, only the ref var r1 redirected
# print(arr1)
# arr1 = np.array([-1.2, 1.8, 2.9])
# rd_arr1 = np.round(arr1).astype(np.float64)
# # print(arr1.astype(int))     # will truncate without round
# # print(arr1.astype(np.int))
# print(rd_arr1)
# print(rd_arr1.dtype)        # int: int32,  float:float64, np.float: float64, np.float64: float64 , print same but not equals
# if int == np.int32:       # np float equals python float, int equals to  but np.int32 not equals to p int.
# 	print('np and python float equals')
# else:
# 	print('not equal')      # but np.float64 not equals to p float.

## cv2 test
# img1 = np.arange(64).reshape([8,8])
# img2 = np.zeros([8,8])
# img2[4,4]=10
# img_rsz = cv2.resize(img2, tuple([1,1]))    # but be tuple
# print(img_rsz)  # 2.5 still have value,so the gaussian will still hold value in there
# img_cv1 = cv2.imread('h36m_samples/s_09_act_02_subact_01_ca_01_000001.jpg')
# print(img1)
# img1=img1.astype(float)
# img1 =  cv2.resize(img1.astype(np.uint8), (256, 256))        # not affecteed
# img1_ds = cv2.resize(img1, (4, 4))  # cv2 float np.uint8 can work, others can't
# img1_ds = cv2.resize(img1.astype(np.float), (4, 4))
# img_cv1_rsz = cv2.resize(img_cv1, (64, 64))
# print(img1_ds)
# print(img1_ds.astype(int))
# cv2.imshow('test', img1.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## matplotlib
# plt.plot([0,1,2], [4, 3, 5])
# plt.show()
# 3d
# Fixing random state for reproducibility
# np.random.seed(19680801)
# def randrange(n, vmin, vmax):
# 	'''
# 	Helper function to make an array of random numbers having shape (n, )
# 	with each number distributed Uniform(vmin, vmax).
# 	'''
# 	return (vmax - vmin) * np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n = 100
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
# 	xs = randrange(n, 23, 32)
# 	ys = randrange(n, 0, 100)
# 	zs = randrange(n, zlow, zhigh)
# 	ax.scatter(xs, ys, zs, marker=m)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# # plt.show()
# plt.savefig(osp.join('output_tmp', 'plt_3d.png'))

# quick check of gened result
# data_files/SURREAL_pm.json
# pth = 'data_files/SURREAL_pm.json'
# with open(pth, 'r') as f:
# 	ds_in = json.load(f)        # 100 samples in
# 	f.close
# print(np.array(ds_in).shape)        # 100 x17 x 3 good
# print('first sample', ds_in[0])

## path operation
input_folder = '/scratch/liu.shu/datasets/SURREAL_demo'
rst = osp.split(input_folder)
print(rst)
