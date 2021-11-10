'''
get the png and generate the raw npy for all extracted images
'''
import shutil
import cv2
import scipy.io as sio
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import os.path as osp


def configure_output_dirs(input_dir, output_dir):
    ''' We want to copy input dir structure to output dir '''
    def ig_f(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    # Don't let this function change an existing folder
    if os.path.exists(output_dir):
        print("ERROR: Output directory already exists!")
        exit(1)

    shutil.copytree(input_dir, output_dir, ignore=ig_f)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--destination_folder', default='/scratch/liu.shu/datasets/SURREAL_demo', help='where to save ds')
parser.add_argument('--rt', type=int, default=1, help='sample rate, original 30')
parser.add_argument('--n_max', type=int, default=10, help='max samples to be collected for demo')
parser.add_argument('--if_exPng', default='n', help='max samples to be collected for demo')
opts, _ = parser.parse_known_args()


rt = opts.rt
n_max = opts.n_max
# destination_folder = '/scratch/sehgal.n/datasets/train_surreal_images'
destination_folder = opts.destination_folder
input_folder = '/scratch/sehgal.n/datasets/surreal/SURREAL/data/cmu/train'
annotation_file_suffix = '_info.mat'
fd_nm = osp.split(destination_folder)[-1]

progress_bar = tqdm(total=211443)

if opts.if_exPng =='y':
    configure_output_dirs(input_folder, destination_folder)
    ct = -1
    for (dirpath, dirnames, filenames) in os.walk(input_folder):
        if len(filenames) > 2:
            for video_file in glob.glob(os.path.join(dirpath, "*.mp4")):
                ct += 1
                if n_max>0 and ct >= n_max:  # only same  limited frames, set 0 for all
                    break
                output_parent_dir = os.path.dirname(video_file.replace(input_folder, destination_folder))
                video_name = os.path.basename(video_file).replace('.mp4', '')
                output_folder = os.path.join(output_parent_dir, video_name)

                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                cap = cv2.VideoCapture(video_file)
                frames_to_read = np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT), rt).astype(int)
                mat_file = os.path.join(dirpath, os.path.basename(video_file).replace('.mp4', '_info.mat'))
                joint_labels = sio.loadmat(mat_file)['joints2D']  # (2, 24, N)

                video_joints = []

                for frame_num in frames_to_read:
                    cap.set(1, frame_num)
                    ret, frame = cap.read()

                    if ret:
                        cv2.imwrite(os.path.join(output_folder, '{}_{:03d}.png'.format(video_name, frame_num)), frame)
                        joints = joint_labels[:, :, frame_num]
                        video_joints.append(joints)
                    progress_bar.update()
                # Save joints
                output_joints_file = os.path.join(output_folder, '{}.npy'.format(video_name))
                np.save(output_joints_file, video_joints)

progress_bar = tqdm(total=211443)
dataset = []
for (dirpath, dirnames, filenames) in os.walk(destination_folder):
    if_hasPng = False
    filenames.sort()        # in place
    for file in filenames:
        if file.endswith('.png'):
            if_hasPng = True
            # Get paths
            # print(file)
            train_folder_path = dirpath.split(fd_nm+'/')[-1]
            vid_file_name = train_folder_path.split('/')[-1]
            surreal_path = os.path.join(input_folder, train_folder_path.split(vid_file_name)[0])

            # Load annotation
            annotation = sio.loadmat(os.path.join(
                surreal_path,
                vid_file_name + annotation_file_suffix))
            frame_num = int(file.split('.png')[0].split('_')[-1])

            # Get annotation for our frame num
            image_name = os.path.join(train_folder_path, vid_file_name + '_{:03d}.png'.format(frame_num))
            assert os.path.exists(os.path.join(destination_folder, image_name)), 'Image doesn\'t exist!'
            this_frame_annotation = {
                'image': image_name,
                'joints2D': annotation['joints2D'][:, :, frame_num],
                'joints3D': annotation['joints3D'][:, :, frame_num],
                'light': annotation['light'][:, frame_num],
                'bg': annotation['bg'][:, frame_num],
                'pose': annotation['pose'][:, frame_num],
                'zrot': annotation['zrot'][frame_num],
                'cloth': annotation['cloth'][:, frame_num],
                'shape': annotation['shape'][:, frame_num],
                'camDist': annotation['camDist'],
            }
            del annotation
            dataset.append(this_frame_annotation)
            progress_bar.update()

# Save and close out
np.save(osp.join(destination_folder, 'surreal_annotations_raw.npy'), dataset)
