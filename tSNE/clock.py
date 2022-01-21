import numpy as np
import glob
import cv2
import os 
import sys
sys.path.append('/scratch/liu.shu/codesPool/AHuP')
from data.dataset import genLoaderFromDs
from opt import opts
from utils.utils_pose import transform_joint_to_other_db
from tqdm import tqdm
import pickle

train_folder = '/scratch/liu.shu/codesPool/AHuP/output/ScanAva-MSCOCO-MPII_res50_n-scraG_10.0D2_n-yl_1rtSYN_regZ0_n-fG_n-nmBone_adam_lr0.001_exp/vis/train'
datasets = ['MuPoTS', 'ScanAva', "SURREAL", 'Human36M']
h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Thorax', 'Neck', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist') # max std joints, first joint_num_ori will be true labeled

## 1. : Get frame numbers for all G_fts_raw in dataset 
feature_frames = {}

for dataset in datasets:
    feature_files = os.listdir(os.path.join(train_folder, dataset, 'G_fts_raw'))
    frame_nums = [int(x.replace('.npy', '')) for x in feature_files]
    feature_frames[dataset] = frame_nums
    print('Got {} frames for dataset {}'.format(len(frame_nums), dataset))
    
## 2. : Get root-centered joint_cam for those frame numbers
for i in range(len(datasets)):
    exec('from data.' + datasets[i] + '.' + datasets[i] + ' import ' + datasets[i])

feature_joints = {}
all_joints = []
all_labels = []
for dataset in datasets:
    dataset_type = ['test' if dataset == 'MuPoTS' else 'train'][0]
    ds = eval(dataset)(dataset_type, opts)

    frame_nums = feature_frames.get(dataset)
    dataset_feature_annos = np.take(ds.data, frame_nums)
    dataset_feature_joints = []

    for i in tqdm(range(len(dataset_feature_annos)), desc=dataset):
        joints = transform_joint_to_other_db(dataset_feature_annos[i]['joint_cam'],
            ds.joints_name, h36m_joints_name)
        joints = joints - joints[0]
        dataset_feature_joints.append([i, joints])

        all_joints.append([joints, dataset, i])

    feature_joints[dataset] = dataset_feature_joints
    del dataset_feature_annos
    del dataset_feature_joints

np.save('all_joints.npy', all_joints)

f = open('feature_joints.pkl', 'wb')
pickle.dump(feature_joints)
f.close()



## 3. : Matching algorithm 

## 4.


## 5.

