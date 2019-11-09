import argparse
from tqdm import tqdm
import os
import scipy.io as sio
import numpy as np

# For ScanAva, we have several constants
ANNOTATION_NAME_2D, ANNOTATION_FILE_2D = 'joints_gt', 'joints_gt.mat'
ANNOTATION_NAME_3D, ANNOTATION_FILE_3D = 'joints_gt3d', 'joints_gt3d.mat'
IMAGE_DIR = 'images'
F = [600, 600]
C = [256, 256]
IMAGE_SHAPE = [512, 512]
ROOT_CAM = [0, 0, 0]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Parent directory of ScanAva data", dest='in_dir',
                        default='/scratch/liu.shu/datasets/ScanAva_1019')
    parser.add_argument('-o', '--output', help='Destination File (must be .npy)', required=True)
    args = parser.parse_args()

    assert args.output.endswith('.npy'), 'Output must be .npy file!'
    return args

def generate_bounding_boxes(joints_2d):
    ''' joints_2d: Nx3x14'''
    # Get corners of bounding box by using the max and min coordinates of the joints
    maxes = np.max(joints_2d, axis=2)[:, :2]  # (N, 2)
    mins = np.min(joints_2d, axis=2)[:, :2]  # (N, 2)
    
    # Pad bounding box to make sure the entire human is in
    margin_y = IMAGE_SHAPE[1] // 10
    margin_x = IMAGE_SHAPE[0] // 10
    maxes = np.add(maxes, [margin_x, margin_y])
    mins = np.add(mins, [margin_x, margin_y])

    # Reshape and return
    bounding_boxes = np.hstack((maxes, mins)).reshape((len(maxes), 2, 2))
    return bounding_boxes

def generate_annotations_for_subfolder(subfolder, parent_dir): 
    subfolder_path = os.path.join(parent_dir, subfolder) 
    
    # Load annotations
    joints_2d = sio.loadmat(os.path.join(subfolder_path, ANNOTATION_FILE_2D))[ANNOTATION_NAME_2D].transpose([2, 0, 1])
    joints_3d = sio.loadmat(os.path.join(subfolder_path, ANNOTATION_FILE_3D))[ANNOTATION_NAME_3D].transpose([2, 0, 1])
    images = sorted(os.listdir(os.path.join(subfolder_path, IMAGE_DIR)))
    paths = [os.path.join(subfolder_path, os.path.join(IMAGE_DIR, x)) for x in images]
    assert len(images) == len(joints_2d) == len(joints_3d), 'Inconsistent number of images and joint annotations'

    # Get bounding boxes
    bounding_boxes = generate_bounding_boxes(joints_2d)

    # Create joints_vis -> set all joints to visible since we do not have occlusion data
    joints_vis = np.ones((len(joints_2d), np.shape(joints_2d)[-1])).astype(int)

    # Format annotations
    this_folder_annotations = []
    labels = ['img_path', 'img_id', 'bbox', 'joint_img', 'joint_cam', 'joint_vis', 'root_cam', 'F', 'C']
    this_folder_annotations = [dict(zip(labels, [path, None, bb, j2d, j3d, jvis, ROOT_CAM, F, C])) for path, bb, j2d, j3d, jvis in zip(paths, bounding_boxes, joints_2d, joints_3d, joints_vis)]
    return this_folder_annotations

if __name__ == '__main__':
    args = get_args()
    parent_dir = args.in_dir
    sub_folders = [sub_folder for sub_folder in os.listdir(parent_dir) if sub_folder.startswith('SYN_') and os.path.isdir(os.path.join(parent_dir, sub_folder))]

    all_annotations = []
    for sub_folder in tqdm(sub_folders):
        all_annotations.append(generate_annotations_for_subfolder(sub_folder, parent_dir)) 
    all_annotations = np.ravel(all_annotations)

    output = args.output
    np.save(output, all_annotations)

