'''
Fundamental utils.
'''
import re
import os
import sys
import torch
import numpy as np
from PIL import Image
import os.path as osp
from . import vis
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_ubyte
import subprocess
import json

#
# def getPCK_single(p1_err,  head_v, dim=3, ref=None):
# 	'''
# 	take the single error array p1_err in square for this implementation. return the PCK_v,
# 	:param p1_err:  N * n_jt * 3  power2
# 	:return:
# 	'''
# 	if not p1_err:
# 		print('warning, empty input')
# 		return -1
# 	# default arguments
# 	if not ref:
# 		if dim==3:
# 			ref = np.array(range(0, 150, 5))
# 		elif dim == 2:
# 			ref = np.array(range(0,1,0.05))
# 	else:
# 		# 3d PCK_vec
# 		pck_v = []
# 		for th in ref:
# 			n_valid = np.sum(np.power(np.sum(p1_err[:dim], axis=2), 0.5)<th)
# 			pck_v.append(float(n_valid)/np.prod(p1_err.size.shape[:2]))
# 		AUC = sum(pck_v)/len(pck_v)
#
# 		return {'pck_v':pck_v, 'AUC':AUC}
#
# def getPCK(p1_err, head_v, dim=3, ref=None):
# 	'''
# 	get PCK result, give PCK3d_v  PCK2d_v, AUC_2d, AUC_3d
# 	:param p1_err:
# 	:return:
# 	'''
# 	if not p1_err:
# 		print('warining, no p1_err input')
# 		return -1
# 	if not ref:
# 		if dim==3:
# 			ref = np.array(range(0, 150, 5))
# 		elif dim == 2:
# 			ref = np.array(range(0,1,0.05))
# 	if type(p1_err) is list:
# 		rst = []
# 		for err_act in p1_err:
# 			rst.append(getPCK_single(err_act, dim=dim, ref=ref))
# 	else:
# 		rst = getPCK_single(p1_err)
# 	return rst

def getPCK_3d(p1_err, ref=tuple(range(0,155,5))):
	'''
	single N x n_jt  distance vec
	:param p1_err:
	:param ref:
	:return:
	'''
	# 3d PCK_vec
	pck_v = []
	for th in ref:
		n_valid = np.sum(p1_err<th)
		pck_v.append(float(n_valid)/p1_err.size)
	auc = sum(pck_v)/len(pck_v)
	return pck_v, auc

def li2str(li):
	'''
	transfer the lsit into a string. right now is for int only
	:param li:
	:return:
	'''
	return ''.join([str(e) for e in li])

def getNumInStr(str_in, tp=int):
	'''
	get the number in list transferred as type(indicated)
	:param str_in: the input string
	:return:
	'''
	temp = re.findall(r'\d+', str_in)
	res = list(map(tp, temp))
	return res


def make_folder(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


def mkdirs(paths):
	"""create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			make_folder(path)
	else:
		make_folder(paths)


def add_pypath(path):
	if path not in sys.path:
		sys.path.insert(0, path)


def tensor2im(input_image, imtype=np.uint8, clipMod='clip01'):
	""""Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor):  # get the data from a variable
			image_tensor = input_image.data
		else:
			return input_image
		image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		if 'clip11' == clipMod:
			image_numpy = (np.transpose(image_numpy,
			                            (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
		else:  # for clip 11 operation
			image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)  # 01 scale directly
	else:  # if it is a numpy array, do nothing
		image_numpy = input_image
	return image_numpy.astype(imtype)       # to H,W,C?


def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	"""Save a numpy image to the disk

	Parameters:
		image_numpy (numpy array) -- input numpy array
		image_path (str)          -- the path of the image
	"""
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def vis_3d(kpt_3d, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id = 1):
	'''
	simplified version with less positional input comparing to vis pack.  Just show the skeleton, if non visibility infor, show full skeleton. Plot in plt, and save it.
	:param kpt_3d:  n_jt * 3
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:param rg: the range for x, y and z in  ( (xs, xe), (ys, ye), (zs, ze)) format
	:return:
	'''

	fig = plt.figure(fig_id)
	ax = fig.add_subplot(111, projection='3d')
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(skel) + 2)]
	colors = [np.array((c[2], c[1], c[0])) for c in colors]

	if not kpt_3d_vis:
		kpt_3d_vis = np.ones((len(kpt_3d), 1))  # all visible

	for l in range(len(skel)):
		i1 = skel[l][0]
		i2 = skel[l][1]
		x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
		y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
		z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

		if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
			ax.plot(x, z, -y, c=colors[l], linewidth=2)
		if kpt_3d_vis[i1, 0] > 0:
			ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[l]], marker='o')
		if kpt_3d_vis[i2, 0] > 0:
			ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[l]], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	if rg:
		ax.set_xlim(rg[0])      # x
		ax.set_zlim([-e for e in rg[1]][::-1])   # - y
		ax.set_ylim(rg[2])

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	# ax.legend()       # no legend
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')
	plt.close(fig)  # clean after use

def vis_3d_cp(kpt_3d_li, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id = 1):
	'''
	visulize the 3d plot in one figure for compare purpose, with differed color
	:param kpt_3d:  n_jt * 3
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:param rg: the range for x, y and z in  ( (xs, xe), (ys, ye), (zs, ze)) format
	:return:
	'''
	if isinstance(kpt_3d_li, np.ndarray):
		kpt_3d_li =[ kpt_3d_li] # to list
	N = len(kpt_3d_li)
	fig = plt.figure(fig_id)
	ax = fig.add_subplot(111, projection='3d')
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, N)]
	colors = [np.array((c[2], c[1], c[0])) for c in colors]
	if not kpt_3d_vis:
		kpt_3d_vis = np.ones((len(kpt_3d_li[0]), 1))  # all visible

	for i, kpt_3d in enumerate(kpt_3d_li):
		for l in range(len(skel)):
			i1 = skel[l][0]
			i2 = skel[l][1]
			x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
			y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
			z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

			if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
				ax.plot(x, z, -y, c=colors[i], linewidth=2)
			if kpt_3d_vis[i1, 0] > 0:
				ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[i]], marker='o')
			if kpt_3d_vis[i2, 0] > 0:
				ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[i]], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	if rg:
		ax.set_xlim(rg[0])      # x
		ax.set_zlim([-e for e in rg[1]][::-1])   # - y
		ax.set_ylim(rg[2])

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	# ax.legend()       # no legend
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')
	plt.close(fig)  # clean after use

def ts2cv2(img_ts, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	'''
	recover the image from tensor to uint8 cv2 fromat with mean and std. Suppose original in 0~1 format. RGB-BGR, cyx -> yxc
	:param img_ts:
	:param mean:
	:param std:
	:return:
	'''
	if not isinstance(img_ts, np.ndarray):  # if tensor transfer it
		tmpimg = img_ts.cpu().detach().numpy()
	else:
		tmpimg = img_ts.copy()
	tmpimg = tmpimg * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
	tmpimg = tmpimg.astype(np.uint8)
	tmpimg = tmpimg[::-1, :, :]  # BGR
	tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()  # x, y , c
	return tmpimg

def showJoints(img, joint_img, svPth = None):
	'''
	label all joints to help figure out joint name
	:param img:
	:param joint_img: n_jt *3 or n_jt *2
	:return:
	'''
	h, w = img.shape[:2]
	offset = 0
	cycle_size = min(1, h/100)
	for i, joint in enumerate(joint_img):
		cv2.circle(img, (joint[0], joint[1]), cycle_size, (0, 255, 0), -1)
		cv2.putText(img, str(i), (joint[0] + offset, joint[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
	if not svPth:
		cv2.imshow('label joints', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		cv2.imwrite(svPth, img)

def save_2d_tg3d(img_patch, pred_2d, skel, sv_dir, idx='tmp'):
	'''
	make joint labeled folder in image, save image into sv_dir/2d/idx.jpg tg3d taskgeneration3d
	:param img_patch: image suppose to be c,w,h rgb numpy
	:param pred_2d: x,y, score  3xn_jt
	:param sv_dir:  where to save
	:return:
	'''
	sv_dir = osp.join(sv_dir, '2d')
	make_folder(sv_dir)
	tmpimg = vis.vis_keypoints(img_patch, pred_2d, skel)
	cv2.imwrite(osp.join(sv_dir, str(idx) + '.jpg'), tmpimg)


def save_3d_tg3d(kpt_3d, sv_dir, skel, idx='tmp', suffix=None):
	'''
	save 3d plot to designated places. sub folder auto generation
	:param coord_out:
	:param sv_dir:
	:param skel:
	:param idx:
	:param suffix:
	:return:
	'''
	rg = None
	if suffix:
		svNm = '3d_' + suffix
		if 'hm' == suffix:
			rg = ((0,64),) * 3
		else:
			rg = ((-1000, 1000), ) * 3
	else:
		svNm = '3d'
	sv_dir = osp.join(sv_dir, svNm)
	make_folder(sv_dir)
	sv_pth = osp.join(sv_dir, str(idx) + '.jpg')
	vis_3d(kpt_3d, skel, sv_pth=sv_pth, rg=rg)


def save_hm_tg3d(HM, sv_dir, n_jt=17, idx='tmp', if_cmap=True):
	'''
	transfer 3d heatmap into front view and side view
	:param HM:  cxhxw  format numpy possibly  0~1  (64x17) * 64 * 64
	:param sv_dir:
	:param idx:
	:return:
	'''
	sv_dir = osp.join(sv_dir, 'hm')
	make_folder(sv_dir)

	# to each jt  # reshape change itself?
	depth_dim = int(HM.shape[0]/n_jt)
	hm = HM.copy().reshape([n_jt, depth_dim, *HM.shape[1:]])
	hm_xy_li = []
	hm_yz_li = []
	for i in range(n_jt):
		hm_xy = hm[i].mean(axis=0)  # channel first
		hm_yz = hm[i].mean(axis=2)  # along the x direction or r direction
		hm_xy_li.append(hm_xy)
		hm_yz_li.append(hm_yz)
		if if_cmap:
			cmap = plt.cm.jet
			norm = plt.Normalize(vmin=hm_xy.min(), vmax=hm_xy.max())
			hm_xy = cmap(norm(hm_xy))
			norm = plt.Normalize(vmin=hm_yz.min(), vmax=hm_yz.max())
			hm_yz = cmap(norm(hm_yz))
		io.imsave(osp.join(sv_dir, 'f{}_jt{}.png'.format(idx, i)), img_as_ubyte(hm_xy))
		io.imsave(osp.join(sv_dir, 's{}_jt{}.png'.format(idx, i)), img_as_ubyte(hm_yz))
	# for total
	hm_xy_tot = np.mean(hm_xy_li, axis=0)
	hm_yz_tot = np.mean(hm_yz_li, axis=0)
	if if_cmap:
		cmap = plt.cm.jet
		norm = plt.Normalize(vmin=hm_xy_tot.min(), vmax=hm_xy_tot.max())
		hm_xy_tot = cmap(norm(hm_xy_tot))
		norm = plt.Normalize(vmin=hm_yz_tot.min(), vmax=hm_yz_tot.max())
		hm_yz_tot = cmap(norm(hm_yz_tot))
	io.imsave(osp.join(sv_dir, 'f{}_tot.png'.format(idx, i)), img_as_ubyte(hm_xy_tot))
	io.imsave(osp.join(sv_dir, 's{}_tot.png'.format(idx, i)), img_as_ubyte(hm_yz_tot))


def save_Gfts_raw_tg3d(G_fts, sv_dir, idx='tmp'):
	'''
	save all G_fts in a raw npy format to for recovery later.
	:param G_fts: already is numpy.
	:param sv_dir_G:
	:param idx:
	:param shape: what grid is needed,  first prod(shape) elements will be used to form grid
	:param out_sz: the output size of the feature map to make it large
	:return:
	'''
	sv_dir_G = osp.join(sv_dir, 'G_fts_raw')
	make_folder(sv_dir_G)
	if type(G_fts) is list:
		for i, G_ft in enumerate(G_fts):
			np.save(osp.join(sv_dir_G, str(idx) + '_' + str(i) + '.npy'), G_fts)    # idx_iLayer.jpg formate
	else:
		np.save(osp.join(sv_dir_G, str(idx)+'.npy'), G_fts) # right now direct idx.npy

def save_Gfts_tg3d(G_fts, sv_dir, idx='tmp', shape=(5, 5), out_sz=(64, 64)):
	'''

	:param G_fts:
	:param sv_dir_G:
	:param idx:
	:param shape: what grid is needed,  first prod(shape) elements will be used to form grid, 5x5 grid of G
	:param out_sz: the output size of the feature map to make it large
	:return:
	'''
	sv_dir_G = osp.join(sv_dir, 'G_fts')
	make_folder(sv_dir_G)
	n = np.prod(shape)
	if type(G_fts) is list:     # for list case
		for i, G_ft in enumerate(G_fts):
			fts = G_ft[:n]  # only first few
			n_cols = shape[1]
			# resize the fts (c last , resize, c back)
			fts_rsz = transform.resize(fts.transpose((1, 2, 0)), out_sz).transpose((2, 0, 1))
			# gallery
			grid = gallery(fts_rsz, n_cols=n_cols)
			#  cmap = plt.cm.jet        # can also color map it
			# save
			norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())
			io.imsave(osp.join(sv_dir_G, str(idx) + '_' + str(i) + '.png'), img_as_ubyte(norm(grid)))

			# for histogram
			sv_dir_hist = osp.join(sv_dir, 'hist')
			make_folder(sv_dir_hist)
			# hist_G = np.histogram(G_fts)
			plt.clf()
			fts_hist = G_fts.flatten()
			fts_hist = fts_hist[fts_hist > 0.1]
			plt.hist(fts_hist, bins=50)
			plt.savefig(osp.join(sv_dir_hist, str(idx) + '_' + str(i) + '.png'))

	else:
		fts = G_fts[:n]  # only first few
		n_cols = shape[1]
		# resize the fts (c last , resize, c back)
		fts_rsz = transform.resize(fts.transpose((1, 2, 0)), out_sz).transpose((2, 0, 1))
		# gallery
		grid = gallery(fts_rsz, n_cols=n_cols)
		#  cmap = plt.cm.jet        # can also color map it
		# save
		norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())
		io.imsave(osp.join(sv_dir_G, str(idx) + '.png'), img_as_ubyte(norm(grid)))

		# for histogram
		sv_dir_hist = osp.join(sv_dir, 'hist')
		make_folder(sv_dir_hist)
		# hist_G = np.histogram(G_fts)
		plt.clf()
		fts_hist = G_fts.flatten()
		fts_hist = fts_hist[fts_hist>0.1]
		plt.hist(fts_hist, bins=50)
		plt.savefig(osp.join(sv_dir_hist, str(idx) + '.png'))

def gallery(array, n_cols=5):
	nindex, height, width = array.shape[:3]
	shp = array.shape
	if len(shp) > 3:
		if_clr = True
		intensity = shp[3]
	else:
		if_clr = False
	nrows = nindex // n_cols
	assert nindex == nrows * n_cols
	# want result.shape = (height*nrows, width*ncols, intensity)
	# shp_new = [nrows,ncols, height, width] + shp[3:]
	if if_clr:
		result = (array.reshape(nrows, n_cols, height, width, intensity)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * n_cols, intensity))
	else:
		result = (array.reshape(nrows, n_cols, height, width)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * n_cols))
	return result


def draw_gaussian(heatmap, center, sigma):
	'''
	will  affect original image
	:param heatmap:
	:param center:
	:param sigma:
	:return:
	'''
	tmp_size = sigma * 3
	mu_x = int(center[0] + 0.5)
	mu_y = int(center[1] + 0.5)
	w, h = heatmap.shape[0], heatmap.shape[1]
	ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # h , w coordinate  left up corner
	br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
	if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:  # so ul coord as height and width
		return heatmap
	size = 2 * tmp_size + 1
	x = np.arange(0, size, 1, np.float32)
	y = x[:, np.newaxis]
	x0 = y0 = size // 2
	g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
	g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
	g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
	img_x = max(0, ul[0]), min(br[0], h)
	img_y = max(0, ul[1]), min(br[1], w)
	try:
		heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
			heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
			g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
	except:
		print('center', center)
		print('gx, gy', g_x, g_y)
		print('img_x, img_y', img_x, img_y)
	return heatmap


def sv_json(rst_dir, pth_head, rst, sv_nm):
	pth = osp.join(rst_dir, '_'.join([pth_head, sv_nm + '.json']))
	print('save result to {}'.format(pth))
	with open(pth, 'w') as f:
		json.dump(rst, f)

def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
		Keys are device ids as integers.
		Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
	"""
	:param model:
	:param input_tensors:
	:param item_length:
	:return:
	"""

	summary = []

	ModuleDetails = namedtuple(
		"Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
	hooks = []
	layer_instances = {}

	def add_hooks(module):

		def hook(module, input, output):
			class_name = str(module.__class__.__name__)

			# if if_multi:        # if mulit output then take first
			#     output = output[0]
			instance_index = 1
			if class_name not in layer_instances:
				layer_instances[class_name] = instance_index
			else:
				instance_index = layer_instances[class_name] + 1
				layer_instances[class_name] = instance_index

			layer_name = class_name + "_" + str(instance_index)

			params = 0

			if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
					class_name.find("Linear") != -1:
				for param_ in module.parameters():
					params += param_.view(-1).size(0)

			flops = "Not Available"
			if class_name.find("Conv") != -1 and hasattr(module, "weight"):
				flops = (
						torch.prod(
							torch.LongTensor(list(module.weight.data.size()))) *
						torch.prod(
							torch.LongTensor(list(output.size())[2:]))).item()
			elif isinstance(module, nn.Linear):
				flops = (torch.prod(torch.LongTensor(list(output.size()))) \
				         * input[0].size(1)).item()

			if isinstance(input[0], list):
				input = input[0]
			if isinstance(output, list) or isinstance(output, tuple):
				output = output[0]  # if return is also a list
			# print(type(output))
			# if isinstance(output, list):    # stg -> tuple(3,)
			#     output = output[0]  # the wht

			summary.append(
				ModuleDetails(  # name tuple
					name=layer_name,
					input_size=list(input[0].size()),
					output_size=list(output.size()),
					num_parameters=params,
					multiply_adds=flops)
			)

		if not isinstance(module, nn.ModuleList) \
				and not isinstance(module, nn.Sequential) \
				and module != model:
			hooks.append(module.register_forward_hook(hook))

	model.eval()
	model.apply(add_hooks)

	space_len = item_length

	model(*input_tensors)
	for hook in hooks:
		hook.remove()

	details = ''
	if verbose:
		details = "Model Summary" + \
		          os.linesep + \
		          "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
			          ' ' * (space_len - len("Name")),
			          ' ' * (space_len - len("Input Size")),
			          ' ' * (space_len - len("Output Size")),
			          ' ' * (space_len - len("Parameters")),
			          ' ' * (space_len - len("Multiply Adds (Flops)"))) \
		          + os.linesep + '-' * space_len * 5 + os.linesep

	params_sum = 0
	flops_sum = 0
	for layer in summary:
		params_sum += layer.num_parameters
		if layer.multiply_adds != "Not Available":
			flops_sum += layer.multiply_adds
		if verbose:
			details += "{}{}{}{}{}{}{}{}{}{}".format(
				layer.name,
				' ' * (space_len - len(layer.name)),
				layer.input_size,
				' ' * (space_len - len(str(layer.input_size))),
				layer.output_size,
				' ' * (space_len - len(str(layer.output_size))),
				layer.num_parameters,
				' ' * (space_len - len(str(layer.num_parameters))),
				layer.multiply_adds,
				' ' * (space_len - len(str(layer.multiply_adds)))) \
			           + os.linesep + '-' * space_len * 5 + os.linesep

	details += os.linesep \
	           + "Total Parameters: {:,}".format(params_sum) \
	           + os.linesep + '-' * space_len * 5 + os.linesep
	details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(
		flops_sum / (1024 ** 3)) \
	           + os.linesep + '-' * space_len * 5 + os.linesep
	details += "Number of Layers" + os.linesep
	for layer in layer_instances:
		details += "{} : {} layers   ".format(layer, layer_instances[layer])

	return details
