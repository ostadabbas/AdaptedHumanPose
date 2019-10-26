import os
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
import utils.utils_pose as ut_p
# from config import cfg


class MPII:
	# todo  add the missing parts to std
	joint_num = 17
	joint_num_ori = 16  # no Torso joint
	joints_name = (
		'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head', 'R_Wrist','R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'Torso')  # max std joints
	joints_eval_name = joints_name[:joint_num_ori]  # no last name
	flip_pairs_name = (
		('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'),
		('R_Shoulder', 'L_Shoulder'), ('R_Elbow', 'L_Elbow'), ('R_Wrist', 'L_Wrist')
	)
	skels_name = (      # original joint skel format
		('Pelvis', 'Thorax'), ('Thorax', 'Neck'), ('Neck', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		('Pelvis', 'R_Hip'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		('Pelvis', 'L_Hip'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)  # for original preferred skel
	flip_pairs = ut_p.nameToIdx(flip_pairs_name, joints_name)
	skeleton = ut_p.nameToIdx(skels_name, joints_name)
	def __init__(self, data_split, opts={}):
		self.data_split = data_split
		self.opts = opts
		# self.img_dir = osp.join('..', 'data', 'MPII')
		self.ds_dir = opts.ds_dir
		self.img_dir = osp.join(opts.ds_dir, 'MPII')  # list name with images
		self.train_annot_path = osp.join(opts.ds_dir, 'MPII', 'annotations', 'train.json')
		# self.flip_pairs = ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13))
		# self.skeleton = (
		# (0, 1), (1, 2), (2, 6), (7, 12), (12, 11), (11, 10), (5, 4), (4, 3), (3, 6), (7, 13), (13, 14), (14, 15),
		# (6, 7), (7, 8), (8, 9))
		self.joints_have_depth = False
		self.data = self.load_data()
		self.if_train = 1 if 'y' == opts.if_ylB else 0  # according to task request to supervise or not the joints.

	def aug_jts(self, jts):
		'''
		augment the jts to add missing part.
		This is specific for each DS according to original joints design
		:return:
		'''
		# todo
		jtNms= self.joints_name
		jt_torso = (jts[jtNms.index('Thorax'),:] + jts[jtNms.index('Pelvis')])/2.   # torso parts
		jt_torso[2] = 1         # assume visible
		jts= np.vstack((jts, jt_torso))
		return jts

	def load_data(self):
		if self.data_split == 'train':
			db = COCO(self.train_annot_path)
		else:
			print('Unknown data subset')
			assert 0

		data = []
		for aid in db.anns.keys():
			ann = db.anns[aid]

			if (ann['image_id'] not in db.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
				continue

			# sanitize bboxes
			x, y, w, h = ann['bbox']
			img = db.loadImgs(ann['image_id'])[0]
			width, height = img['width'], img['height']
			x1 = np.max((0, x))
			y1 = np.max((0, y))
			x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
			y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
			if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
				bbox = np.array([x1, y1, x2 - x1, y2 - y1])
			else:
				continue

			# aspect ratio preserving bbox
			w = bbox[2]
			h = bbox[3]
			c_x = bbox[0] + w / 2.
			c_y = bbox[1] + h / 2.
			aspect_ratio = self.opts.input_shape[1] / self.opts.input_shape[0]
			if w > aspect_ratio * h:
				h = w / aspect_ratio
			elif w < aspect_ratio * h:
				w = h * aspect_ratio
			bbox[2] = w * 1.25
			bbox[3] = h * 1.25
			bbox[0] = c_x - bbox[2] / 2.
			bbox[1] = c_y - bbox[3] / 2.

			# joints and vis
			joint_img = np.array(ann['keypoints']).reshape(self.joint_num_ori, 3)
			# add missing torso part
			joint_img = self.aug_jts(joint_img)
			# joint_vis = joint_img[:, 2].copy().reshape(-1, 1)
			joint_vis = joint_img[:, 2].copy().reshape(-1, 1) * self.if_train   # if not train, all not visible
			joint_img[:, 2] = 0

			imgname = db.imgs[ann['image_id']]['file_name']
			img_path = osp.join(self.img_dir, imgname)
			data.append({
				'img_path': img_path,
				'bbox': bbox,
				'joint_img': joint_img,  # [org_img_x, org_img_y, 0]
				'joint_vis': joint_vis,
			})

		return data
