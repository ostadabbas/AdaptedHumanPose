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
from skimage import io, transform


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


def tensor2im(input_image, imtype=np.uint8, clipMod='clip11'):
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
	return image_numpy.astype(imtype)


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


def vis_3d(kpt_3d, skel, kpt_3d_vis=None, sv_pth=None):
	'''
	simplified version comparing to vis pack.  Just show the skeleton, if non visibility infor, show full skeleton. Plot in plt, and save it.
	:param kpt_3d:
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:return:
	'''

	fig = plt.figure()
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
			ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
		if kpt_3d_vis[i2, 0] > 0:
			ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	ax.legend()
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')


def ts2cv2(img_ts, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	'''
	recover the image from tensor to uint8 cv2 fromat with mean and std. Suppose original in 0~1 format. RGB-BGR, cyx -> yxc
	:param img_ts:
	:param mean:
	:param std:
	:return:
	'''
	tmpimg = img_ts.cpu().numpy()
	tmpimg = tmpimg * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
	tmpimg = tmpimg.astype(np.uint8)
	tmpimg = tmpimg[::-1, :, :]  # BGR
	tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()  # x, y , c
	return tmpimg


def save_2d_tg3d(img_patch, pred_2d, skel, sv_dir, idx='tmp'):
	'''
	make joint labeled folder in image, save image into sv_dir/2d/idx.jpg
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
	save 3d plot to designated places.
	:param coord_out:
	:param sv_dir:
	:param skel:
	:param idx:
	:param suffix:
	:return:
	'''
	if suffix:
		svNm = '3d_' + suffix
	else:
		svNm = '3d'
	sv_dir = osp.join(sv_dir, svNm)
	make_folder(sv_dir)
	sv_pth = osp.join(sv_dir, str(idx) + '.jpg')
	vis_3d(kpt_3d, skel, sv_pth=sv_pth)


def save_hm_tg3d(HM, sv_dir, idx='tmp', if_cmap=True):
	'''
	transfer 3d heatmap into front view and side view
	:param HM:  cxhxw  format numpy possibly  0~1
	:param sv_dir:
	:param idx:
	:return:
	'''
	sv_dir = osp.join(sv_dir, 'hm')
	make_folder(sv_dir)

	hm_xy = HM.mean(axis=0)  # channel first
	hm_yz = HM.mean(axis=2)  # along the x direction or r direction
	if if_cmap:
		cmap = plt.cm.jet
		norm = plt.Normalize(vmin=hm_xy.min(), vmax=hm_xy.max())
		hm_xy = cmap(norm(hm_xy))
		norm = plt.Normalize(vmin=hm_yz.min(), vmax=hm_yz.max())
		hm_yz = cmap(norm(hm_yz))

	io.imsave(osp.join(sv_dir, 'f_' + str(idx) + '.jpg'), hm_xy)
	io.imsave(osp.join(sv_dir, 's_' + str(idx) + '.jpg'), hm_yz)


def save_Gfts_tg3d(G_fts, sv_dir, idx='tmp', shape=(5, 5), out_sz=(64, 64)):
	'''

	:param G_fts:
	:param sv_dir:
	:param idx:
	:param shape: what grid is needed,  first prod(shape) elements will be used to form grid
	:param out_sz: the output size of the feature map to make it large
	:return:
	'''
	sv_dir = osp.join(sv_dir, 'G_fts')
	make_folder(sv_dir)
	n = np.prod(shape)
	fts = G_fts[:n]  # only first few
	n_cols = shape(1)
	# resize the fts (c last , resize, c back)
	fts_rsz = transform.resize(fts.transpose((1, 2, 0)), out_sz).transpose((2, 0, 1))
	# gallery
	grid = gallery(fts_rsz, n_cols=n_cols)
	#  cmap = plt.cm.jet        # can also color map it
	# save
	io.imsave(osp.join(sv_dir, str(idx)+'.jpg'), grid)

	# for histogram
	sv_dir = osp.join(sv_dir, 'hist')
	make_folder(sv_dir)
	# hist_G = np.histogram(G_fts)
	plt.hist(G_fts, bins=50)
	plt.savefig(osp.join(sv_dir, str(idx)+'.jpg'))

def gallery(array, ncols=5):
	nindex, height, width = array.shape[:3]
	shp = array.shape
	if len(shp) > 3:
		if_clr = True
		intensity = shp[3]
	else:
		if_clr = False
	nrows = nindex // ncols
	assert nindex == nrows * ncols
	# want result.shape = (height*nrows, width*ncols, intensity)
	# shp_new = [nrows,ncols, height, width] + shp[3:]
	if if_clr:
		result = (array.reshape(nrows, ncols, height, width, intensity)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * ncols, intensity))
	else:
		result = (array.reshape(nrows, ncols, height, width)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * ncols))
	return result
