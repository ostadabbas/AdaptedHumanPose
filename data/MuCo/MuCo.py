import os
import os.path as osp
import numpy as np
import math
from utils.utils_pose import get_bbox
from pycocotools.coco import COCO
from utils.evaluate import evaluate

class MuCo:  # we don't use this one, so not processed
    if_SYN = False
    joints_have_depth = True
    # joints_name = (
    # 'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
    # 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')  #  original the head top and head is not same as h36m template
    joints_name = (
    'Head', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
    'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Neck', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')  # what is head and head top
    boneLen2d_av_mm = 3600  # MuPoTs one need to recalculate if needed
    def __init__(self, data_split, opts={}):
        self.data_split = data_split
        self.ds_dir = opts.ds_dir
        # self.img_dir = osp.join('..', 'data', 'MuCo', 'data') # data is redundant
        self.img_dir = osp.join(opts.ds_dir, 'MuCo')
        self.train_annot_path = osp.join(opts.ds_dir, 'MuCo', 'MuCo-3DHP.json')
        self.joint_num = 21
        self.opts = opts

        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):
        # if self.data_split == 'train':      # only train part avaialbe
        #     db = COCO(self.train_annot_path)
        # else:
        #     print('Unknown data subset')
        #     assert 0
        db = COCO(self.train_annot_path)        # all session can use this

        data = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            f = img["f"]
            c = img["c"]

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            root_depths = [ann['keypoints_cam'][self.root_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]
            for i in range(len(anns)):
                if i == closest_pid:
                    continue
                picked = True
                for j in range(len(anns)):
                    if i == j:
                        continue
                    dist = (np.array(anns[i]['keypoints_cam'][self.root_idx]) - np.array(anns[j]['keypoints_cam'][self.root_idx])) ** 2
                    dist_2d = math.sqrt(np.sum(dist[:2]))
                    dist_3d = math.sqrt(np.sum(dist))
                    if dist_2d < 500 or dist_3d < 500:
                        picked = False
                if picked:
                    pid_list.append(i)
            
            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                root_cam = joint_cam[self.root_idx]
                
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_img[:,2] = joint_img[:,2] - root_cam[2]
                joint_vis = np.ones((self.joint_num,1))

                bbox = np.array(anns[pid]['bbox'])

                # sanitize bboxes
                x, y, w, h = bbox
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
                if w*h > 0 and x2 > x1 and y2 > y1: # set strict larger?
                    bbox = np.array([x1, y1, x2-x1, y2-y1])
                else:
                    continue

                # aspect ratio preserving bbox
                w = bbox[2]
                h = bbox[3]
                c_x = bbox[0] + w/2.
                c_y = bbox[1] + h/2.
                aspect_ratio = self.opts.input_shape[1]/self.opts.input_shape[0]
                if w > aspect_ratio * h:
                    h = w / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio
                bbox[2] = w*1.25
                bbox[3] = h*1.25
                bbox[0] = c_x - bbox[2]/2.
                bbox[1] = c_y - bbox[3]/2.

                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'joint_vis': joint_vis,
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c
                })


        return data

    def evaluate(self, preds, **kwargs):
        '''
		rewrite this one. preds follow opts.ref_joint,  gt transfer to ref_joints, taken with ref_evals_idx. Only testset will calculate the MPJPE PA to prevent the SVD diverging during training.
		:param preds: xyz HM
		:param kwargs:  jt_adj, logger_test, if_svEval,  if_svVis
		:return:
		'''
        jt_adj = kwargs.get('jt_adj', None)
        logger_test = kwargs.get('logger_test', None)
        if_svEval = kwargs.get('if_svEval', False)
        if_svVis = kwargs.get('if_svVis', False)
        pth_head = kwargs.get('pth_hd')

        print('Evaluation start...')
        gts = self.data
        assert (len(preds) <= len(gts))  # can be smaller, preds_hm!!

        if self.data_split == 'test':
            if_align = True
        else:
            if_align = False  # for slim evaluation

        if logger_test:
            prt_func = logger_test.info
        else:
            prt_func = print
        evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=None, fn_getIdx=None,
                 opts=self.opts, if_svVis=if_svVis, pth_head=pth_head, fn_prt=prt_func)
