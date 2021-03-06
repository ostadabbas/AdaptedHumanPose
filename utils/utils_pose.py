import torch
import numpy as np
import copy
from collections import OrderedDict
from tqdm import tqdm


def cam2pixel(cam_coord, f, c):
	x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
	y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
	z = cam_coord[..., 2]

	return x, y, z


def pixel2cam(pixel_coord, f, c):
	x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
	y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
	z = pixel_coord[..., 2]

	return x, y, z


def world2cam(world_coord, R, T):
	cam_coord = np.dot(R, world_coord - T)
	return cam_coord


def rigid_transform_3D(A, B):  # solve sum(R*A + T - B)^2 smallest
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
	U, s, V = np.linalg.svd(H)
	R = np.dot(np.transpose(V), np.transpose(U))
	if np.linalg.det(R) < 0:
		V[2] = -V[2]
		R = np.dot(np.transpose(V), np.transpose(U))
	t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
	return R, t


def rigid_align(A, B):
	R, t = rigid_transform_3D(A, B)
	A2 = np.transpose(np.dot(R, np.transpose(A))) + t
	return A2


def get_bbox(joint_img):
	# bbox extract from keypoint coordinates
	bbox = np.zeros((4))
	xmin = np.min(joint_img[:, 0])
	ymin = np.min(joint_img[:, 1])
	xmax = np.max(joint_img[:, 0])
	ymax = np.max(joint_img[:, 1])
	width = xmax - xmin - 1
	height = ymax - ymin - 1

	bbox[0] = (xmin + xmax) / 2. - width / 2 * 1.2
	bbox[1] = (ymin + ymax) / 2. - height / 2 * 1.2
	bbox[2] = width * 1.2
	bbox[3] = height * 1.2

	return bbox


# def warp_coord_to_original(joint_out, bbox, center_cam):
#
#     # joint_out: output from soft-argmax, x,y (pix:oriImg)  z[mm: cam]
#     x = joint_out[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
#     y = joint_out[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
#     z = (joint_out[:, 2] / cfg.depth_dim * 2. - 1.) * (cfg.bbox_3d_shape[0]/2.) + center_cam[2]
#     return x, y, z


def warp_coord_to_ori(joint_out, bbox, center_cam, boneLen2d_mm=3800, opts={}, skel=None):
	'''
	From output joint: HM, bb, camera center, recover to cam coordinate x,y pix:ori z mm: cam
	:param joint_out:
	:param bbox:
	:param center_cam:
	:param boneLen2d_mm:
	:param opts: depth_dim  output_shape, if_normBone,
	:param skel: the skeleton index format, to get boneLen2d_pix
	:return:
	'''
	# joint_out: output from soft-argmax, x,y (pix:oriImg)  z[mm: cam]
	x = joint_out[:, 0] / opts.output_shape[1] * bbox[2] + bbox[0]
	y = joint_out[:, 1] / opts.output_shape[0] * bbox[3] + bbox[1]
	z_unit = (joint_out[:, 2] / opts.depth_dim * 2. - 1.)  # -1 ~ 1  could be around 0.5 if square
	if 'y' == opts.if_normBone and skel:
		boneLen2d_pix = get_boneLen(joint_out[:, 0:2], skel)  # skel is usaully opts.ref_skel, in case we would like to recover to ds order,  in hm
		z = z_unit * opts.output_shape[0] / boneLen2d_pix * boneLen2d_mm + center_cam[2]
	else:
		z = z_unit * opts.bbox_3d_shape[0] / 2. + center_cam[2]
	# z = (joint_out[:, 2] / cfg.depth_dim * 2. - 1.) * (cfg.bbox_3d_shape[0] / 2.) + center_cam[2]
	return x, y, z

def nameToIdx(name_tuple, joints_name):     # test, tp,
	'''
	from reference joints_name, change current name list into index form
	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
	:param joints_name:
	:return:
	'''
	jtNm = joints_name
	if type(name_tuple[0]) == tuple:
		# Transer name_tuple to idx
		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
	else:
		# direct transfer
		return tuple(jtNm.index(tpl) for tpl in name_tuple)

def transform_joint_to_other_db(src_joint, src_name, dst_name):
	'''
	loop source name, find index in new, assign value to that position
	:param src_joint:
	:param src_name:
	:param dst_name:
	:return:
	'''
	src_joint_num = len(src_name)
	dst_joint_num = len(dst_name)

	new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

	for src_idx in range(len(src_name)):  # all 0, then ori name in tar index, assign value
		name = src_name[src_idx]
		if name in dst_name:
			dst_idx = dst_name.index(name)
			new_joint[dst_idx] = src_joint[src_idx]

	return new_joint


def fliplr_joints(_joints, width, matched_parts):
	"""
	flip coords
	joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
	width: image width
	matched_parts: list of pairs
	"""
	joints = _joints.copy()
	# Flip horizontal
	joints[:, 0] = width - joints[:, 0] - 1

	# Change left-right parts
	for pair in matched_parts:
		joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

	return joints


def multi_meshgrid(*args):
	"""
	Creates a meshgrid from possibly many
	elements (instead of only 2).
	Returns a nd tensor with as many dimensions
	as there are arguments
	"""
	args = list(args)
	template = [1 for _ in args]
	for i in range(len(args)):
		n = args[i].shape[0]
		template_copy = template.copy()
		template_copy[i] = n
		args[i] = args[i].view(*template_copy)
		# there will be some broadcast magic going on
	return tuple(args)


def flip(tensor, dims):
	if not isinstance(dims, (tuple, list)):
		dims = [dims]
	indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
	                        dtype=torch.int64) for dim in dims] # ts shape[dim], flip
	multi_indices = multi_meshgrid(*indices)
	final_indices = [slice(i) for i in tensor.shape]
	for i, dim in enumerate(dims):
		final_indices[dim] = multi_indices[i]
	flipped = tensor[final_indices]
	assert flipped.device == tensor.device
	assert flipped.requires_grad == tensor.requires_grad
	return flipped


def get_boneLen(joints, skeleton):
	'''
	get the bone length in mm based on the joints and skeleton definition. These can be gotten from ds definition. for 2d,3d both.
	:param joints:
	:param skeleton:
	:return:
	'''
	s = 0
	for e in skeleton:
		s += ((joints[e[0]] - joints[e[1]]) ** 2).sum() ** 0.5
	return s

def get_boneLen_av(annos, skel, dim=2, fn_getIdx= None, jtNm='joint_cam'):
	'''
	get the average bone length, based on the skels provided. though flexible design, you suppose to use eval joint to form skel which is can be gotten via  ut_p.idx2Nm func. provide average bone len for each. If no fn_getIdx, then return only the sum result.
	:param annos:
	:param skels:
	:param dim: dim depth for bone calculation
	:return:
	'''
	boneSum_dict = OrderedDict()  # keep sum of each subject
	n_dict = OrderedDict()
	bone_sum = 0.
	N = len(annos)
	for anno in tqdm(annos):
		img_path = anno['img_path']  # eg: s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg
		joints_cam = anno[jtNm]
		boneLen = get_boneLen(joints_cam[:, :dim], skel)
		bone_sum += boneLen
		if fn_getIdx:
			idx_subj = fn_getIdx(img_path)
			if idx_subj in boneSum_dict:
				boneSum_dict[idx_subj] += boneLen
				n_dict[idx_subj] += 1
			else:  # first
				boneSum_dict[idx_subj] = boneLen
				n_dict[idx_subj] = 1
	bone_av = float(bone_sum)/N
	if fn_getIdx:
		for k in boneSum_dict:
			boneSum_dict[k] = float(boneSum_dict[k]) / n_dict[k]
	return bone_av, boneSum_dict

def get_l_skels(jts, skels, skel_norm_pair):
	'''
	from the jts, calculate the normrlized skel length.  skels give sthe list of  idx  paris of skels.
	skel_norm gives the idx pair of normallized  skel jt.
	:param jts:
	:param skels:
	:param skel_norm_pair: the joint pair for normalization skeleton
	:return: return the skels vector as len(skels)
	'''
	skel_norm = np.linalg.norm(jts[skel_norm_pair[0]] - jts[skel_norm_pair[1]])
	skels = np.array(skels)     #  N x  2
	l_skels = np.linalg.norm(jts[skels[:,0]] - jts[skels[:,1]], axis=1)/skel_norm
	return l_skels

def get_l_skels_tch(jts, skels, skel_norm_pair=None):
	'''
	from the jts, calculate the normrlized skel length.  skels give sthe list of  idx  paris of skels.
	skel_norm gives the idx pair of normallized  skel jt. the tensor version, with the batch leading dim.
	:param jts:
	:param skels:
	:param skel_norm_pair: the joint pair for normalization skeleton
	:return: return the skels vector as len(skels)
	'''

	skel_norm = (jts[:, skel_norm_pair[0]] - jts[:, skel_norm_pair[1]]).norm(dim=-1).view(-1, 1)     # the N x 1 vector
	skels = np.array(skels)
	l_skels = (jts[:, skels[:,0]] - jts[:, skels[:, 1]]).norm(dim=-1)/skel_norm     # N x  n_skel
	return l_skels      # N x l_skel

