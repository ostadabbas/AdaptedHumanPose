'''
Fundamental utils.
'''
import re
import os
import sys
import torch
import numpy as np
from PIL import Image


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
			image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
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
