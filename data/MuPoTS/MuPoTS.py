import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
import json
import cv2
import random
import math
from utils.utils_pose import pixel2cam, get_bbox, warp_coord_to_ori
from utils.evaluate import evaluate, prt_rst
import utils.utils_tool as ut_t
from utils.vis import vis_keypoints, vis_3d_skeleton

class MuPoTS:
    if_SYN = False
    joints_have_depth = True
    # use uniform name for coding generalization convenience actually subj here
    action_names = ['TS{}'.format(i+1) for i in range(20)]   # TS 1 ~ 20
    boneLen2d_av_mm = 3600
    # missing parts: avBone
    @staticmethod
    def getNmIdx(pth):
        rel_nm = pth.split('MultiPersonTestSet')[-1][1:]  # get rid of the slash
        nums = ut_t.getNumInStr(rel_nm)
        return nums[0]-1  # idx from 0.  subj idx from 1

    def __init__(self, data_split, opts={}):
        self.data_split = data_split
        self.ds_dir = opts.ds_dir
        self.opts = opts
        self.img_dir = osp.join(opts.ds_dir, 'MuPoTS', 'MultiPersonTestSet')
        # self.test_annot_path = osp.join('..', 'data', 'MuPoTS', 'data', 'MuPoTS-3D.json') # there is data sub originally
        self.test_annot_path = osp.join(opts.ds_dir, 'MuPoTS', 'MuPoTS-3D.json')
        self.human_bbox_root_dir = osp.join(opts.ds_dir, 'MuPoTS', 'bbox_root', 'bbox_root_mupots_output.json')
        self.joint_num = 21 # MuCo-3DHP  seems to be annotation saved
        # self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') # MuCo-3DHP original
        self.joints_name = ('Head', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Neck', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') # MuCo-3DHP
        self.original_joint_num = 17 # MuPoTS
        self.original_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head') # MuPoTS, Thorax is like neck

        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7) )
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):
        
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        data = []
        db = COCO(self.test_annot_path)

        # use gt bbox and root
        if self.opts.use_gt_info:
            print("Get bounding box and root from groundtruth")
            for aid in db.anns.keys():
                ann = db.anns[aid]
                if ann['is_valid'] == 0:
                    continue

                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);

                joint_cam = np.array(ann['keypoints_cam'])
                root_cam = joint_cam[self.root_idx]

                joint_img = np.array(ann['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_img[:,2] = joint_img[:,2] - root_cam[2]
                joint_vis = np.ones((self.original_joint_num,1))

                bbox = np.array(ann['bbox'])
                img_width, img_height = img['width'], img['height']

                # sanitize bboxes
                x, y, w, h = bbox
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
                if w*h > 0 and x2 >= x1 and y2 >= y1:
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
                    'c': c,
                })
           
        else:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            
            for i in range(len(annot)):
                image_id = annot[i]['image_id']
                img = db.loadImgs(image_id)[0]
                img_width, img_height = img['width'], img['height']
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);
                root_cam = np.array(annot[i]['root_cam']).reshape(3)
                bbox = np.array(annot[i]['bbox']).reshape(4)

                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_cam': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_vis': np.zeros((self.original_joint_num, 1)), # dummy
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c,
                })

        return data


    def evaluate(self, preds, **kwargs):
        '''
		rewrite this one. preds follow opts.ref_joint,  gt transfer to ref_joints, taken with ref_evals_idx. Only testset will calculate the MPJPE PA to prevent the SVD diverging during training.
		:param preds: xyz HM
		:param kwargs:  jt_adj, logger_test, if_svEval,  if_svVis
		:return:
		'''
        logger_test = kwargs.get('logger_test', None)
        if_svEval = kwargs.get('if_svEval', False)
        if_svVis = kwargs.get('if_svVis', False)
        print('Evaluation start...')
        gts = self.data
        assert (len(preds) <= len(gts))  # can be smaller
        # gts = gts[:len(preds)]  # take part of it
        # joint_num = self.joint_num
        if self.data_split == 'test':
            if_align = True
        else:
            if_align =False     # for slim evaluation

        # all can be wrapped together  ...
            # name head, ds specific
        if if_svEval:
            pth_head = '_'.join([self.opts.nmTest, self.data_split])  # ave bone only in recover not affect the HM

        else:
            pth_head = None
            # get prt func
        if logger_test:
            prt_func = logger_test.info
        else:
            prt_func = print

        evaluate(preds, gts, self.joints_name, if_align=if_align, act_nm_li=self.action_names, fn_getIdx=self.getNmIdx, opts=self.opts, avBone=self.boneLen2d_av_mm, if_svVis=if_svVis, pth_head=pth_head, fn_prt=prt_func)

    def evaluate1(self, preds, result_dir):  # only save mat result here
        # original evaluate
        print('Evaluation start...')
        gts = self.data
        sample_num = len(preds)
        joint_num = self.original_joint_num
        # if opts.ref_joint_name != self.joint_name
        pred_2d_save = {}
        pred_3d_save = {}
        for n in range(sample_num):
            
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            img_name = gt['img_path'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0] # e.g., TS1_img_0001
            
            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            # only consider eval_joint
            pred_2d_kpt = np.take(pred_2d_kpt, self.eval_joint, axis=0)
            # pred_2d_kpt[:,0], pred_2d_kpt[:,1], pred_2d_kpt[:,2] = warp_coord_to_original(pred_2d_kpt, bbox, gt_3d_root)
            pred_2d_kpt[:,0], pred_2d_kpt[:,1], pred_2d_kpt[:,2] = warp_coord_to_ori(pred_2d_kpt, bbox, gt_3d_root, opts=self.opts, skel=self.opts.ref_skels_idx)

            # 2d kpt save
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:,:2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:,:2]]

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1,500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3,joint_num))
                tmpkps[0,:], tmpkps[1,:] = pred_2d_kpt[:,0], pred_2d_kpt[:,1]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = np.zeros((joint_num,3))
            pred_3d_kpt[:,0], pred_3d_kpt[:,1], pred_3d_kpt[:,2] = pixel2cam(pred_2d_kpt, f, c)
            
            # 3d kpt save
            if img_name in pred_3d_save:
                pred_3d_save[img_name].append(pred_3d_kpt)
            else:
                pred_3d_save[img_name] = [pred_3d_kpt]
        
        output_path = osp.join(result_dir,'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join(result_dir,'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)

