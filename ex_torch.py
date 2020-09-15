'''
test the functionalities of pytorch
'''
import opt
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import itertools
from torch.nn import ModuleList as ML
from torch import nn

# opts = opt.opts

## basics
# ts1 = torch.Tensor([1,2,3])
# ts1= torch.Tensor([True] * 5)      # all to float, 1.0
# ts1= torch.tensor([True] * 5, dtype=bool)       # Capital not working , key words error
# ts1 = torch.tensor(0.2)       # can be translated to float directly
# ts1 = torch.tensor([0.2, 0.3]).cuda()  # 2 can't translate to float
ts1 = torch.ones([5, 3, 3,3]).cuda()
arr1 = np.ones(3)
# conv = torch.nn.Conv2d(3, 1, kernel_size=3).cuda()     # 5  x 1
print('is tensor', isinstance(ts1, torch.Tensor))
print('is np.ndarray', isinstance(arr1, np.ndarray))
# print(conv(ts1) )       # simple conv
# print(float(ts1))
# print(ts1, 'shape', ts1.shape)
# ts1 = torch.ones(2)
# ts2 = torch.ones([2,])      # ones can have multiple int  or  list
# ts1 = torch.ones([3,3, 2, 2, ])
# ts2 = torch.ones([10, 1])
# ts1m2 = 1 - ts2
# print(ts1m2)        # keep the  dim  10x1
# print(ts1m2.shape)
# print(ts2.view(-1)) #
# print(ts2.view(-1).shape)
# ts_idx1 = torch.Tensor([1, 0, 1]).type(torch.bool)       # to  False True thing
# ts_idx1 = torch.BoolTensor([1, 0, 1])       # use int to index?
# print(ts_idx1 == 1)     # tensor ([True, False, True])
# rst = ts1[ts_idx1==1]       # can use single dim index
# print('ts_idx', ts_idx1)
# rst = ts1[ts_idx1]       # can use single dim index
# print(rst, 'shape', rst.shape)
# print(ts1[ts_idx1==1])      # two tensor gotten
# ts2 = torch.Tensor([3,8,2])
# arr1 = np.array([1,2,3])
# li1 = [ts1, ts2]
#
# print(sum(li1))
# print(not li1)      # return false
# print(not ts1)      # more than one value is ambiguous
# print(not arr1)
# ts1 = torch.ones([2, 1,3, 4])     # N x3 x4
# print(ts1.shape)        # 2,3, 4  ts1 size the same
# print(ts1.size())
# ts2 = torch.zeros([2, 1, 3, 4])
# print(nn.MSELoss(reduction='none')(ts1, ts2))   # no reduction is 1  average over all  original size
# if ts1:       # can't determine an array with mulitple elements in if
# 	print('we can determin if a tensor exist')
# else:
# 	print("tensor equals to None or 0 in if")
# m = nn.LogSoftmax()
# m1 = nn.Linear(3,3)
# l_tot = torch.FloatTensor()
# ts1 = torch.Tensor(0)       #  empty
# ts1 = torch.Tensor(0.5)       #  empty
# ts2 = torch.tensor()      # error must have data
# ts1 = torch.Tensor()     #    empty shape, add now no values.
# ts1 = 0.        # this is the right way to keep the loss
# print('ts11', ts1, 'shape', ts1.shape)      # shape 1
# print(l_tot)        # empty nothing
# input = torch.randn(2, 3)       # require false
# output = m(m1(input))
# l = nn.NLLLoss()(output, torch.tensor([1,0]))  # bch size  should be 2 , all loss add to one value
# print(output, 'shape is', output.shape) # same size neg inf to 0,  same size so the batch kept
# l_tot = l + l_tot      # if add tensor requires grad, auto requires， still empty
# ts1 += l
# l.backward()          # require grad true
# print(l, 'shape', l.shape)
# print('ts1', ts1, 'shape', ts1.shape)
# shape operation
# ts1  = torch.arange(3).view(3,1)
# ts2 = torch.ones([2,4])
# print(ts2.numel())
# sum1 = ts1.sum()
# print(ts2, 'shape', ts2.shape)      # tensor 2 no shape
# print(sum1, 'shape', sum1.shape)        # tensor 3, no size
# ts2 = torch.ones([3,1,5,5])
# ts2 = torch.ones([3,1])
# shp = ts1.shape
# shp = [-1] + [1]*(ts2.ndimension()-1)
# print(shp)      # should be ts size
# print(*shp)
# print(*shp)
# shp[1:]=1       # size not support item assignment
# print(shp)
# ts1_2 = ts1.view(*shp)
# print(ts1_2,'shape', ts1_2.shape)


## transform
# transform = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean=(0.4,), std=(0.4,))])
# transform = transforms.Compose([
# 	transforms.ToTensor(),
# 	])
#
# # img_cv1 = cv2.imread('h36m_samples/s_09_act_02_subact_01_ca_01_000001.jpg')
# img1 = np.arange(64).reshape([8, 8])    # will add another dimension for channel
# # img_ts1 = transform(img1) # add one dim
# # img_ts1 = torch.tensor(img1)  # save size
# img_ts1 = transforms.ToTensor()(img1)       #  1,8,8
# # # print(type(img_ts1))        # tensor
# print(img_ts1.size())       # 3 1002 100
# arr1 = img_ts1.numpy()      # add one dim
# print(arr1.shape)              # transfor channel dim add to  1,8,8;  torch.tensor(arr)  same dim
# arr1 = np.zeros([1, 2, 3])  # 1 dim not changed
# arr2 = np.array([1.2, 2.3])
# arr1[0,2] = 1
# print(arr1.dtype)
# arr1 = np.zeros_like(arr2)
# print(arr1.dtype)
# ts_arr1 = transforms.ToTensor()(arr1)
# print(ts_arr1)  # no change, channel dimension, 1 still one.  [1,2,3] to 3x1x2 ,  [2,2,3] to 3x2x2, swap axes
# arr1 = np.zeros(2, *[3, 4])
# print(arr1)

## dataloader test
from torch.utils.data import Dataset

#
# class NumbersDataset(Dataset):
# 	def __init__(self):
# 		self.samples = list(range(1, 1001))
#
# 	def __len__(self):
# 		return len(self.samples)
#
# 	def __getitem__(self, idx):
# 		return self.samples[idx]
#
# if __name__ == '__main__':
# 	dataset = NumbersDataset()
# 	print(len(dataset))
# 	print(dataset[100])
# 	print(dataset[122:361])

## network module
# nt_l1 = torch.nn.Linear(2, 2)
# print(nt_l1)
# print(nt_l1.weight)
# li_para = [nt_l1.parameters()]        # only generator obj
# li_para = [p for p in nt_l1.parameters()] # only tensor like parameter, not  name
# print(li_para)      # generator object
# print(nt_l1.state_dict())   # name and weights altogether  {'weights':ts, 'bias':ts}
# li_md = torch.nn.ModuleList([torch.nn.ModuleList([nt_l1 for i in range(3)])for j in range(2)])  # ass 3 list
# print(li_md.state_dict())   #  all saved in to i.j.bias i.j.weights for mli2 array (module list 2d array)
# li_nt_l1 = [nt_l1] * 5  # reference modify one will affect other
# li_nt_l1 = itertools.repeat(nt_l1, 5)   # module can't work here
# li_str = itertools.repeat('mama', 5)    # this can work not module
# li_nt_l1[1].weight[0,  0] = 20
# print(li_nt_l1[2].weight) # affected
# print(isinstance(li_md, list))  # not list
# BCE input shape
# m1 = nn.Linear(10,1)        # output is single dime so  -> N x 1
# ts1 = torch.randn([5,10])
# c1 = nn.BCEWithLogitsLoss()
# print(ts1.shape)
# rst = m1(ts1)
# l = c1(rst, torch.ones([5,]))       # can't work  output  5x1   tar 5 , must 5x1 both
# print(rst, 'shape',rst.shape)
# print(l, 'shape', l.shape)

## criterion
# ts1 = torch.Tensor([2,3,5])
# ts2 = torch.Tensor([1, 6, 9 ])
# f_loss = torch.nn.MSELoss(reduction='none')     # no reduction happens
# print(f_loss(ts1, ts2))

