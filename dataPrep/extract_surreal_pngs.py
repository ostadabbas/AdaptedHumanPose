'''
get the png
'''
import shutil
import cv2
import scipy.io as sio
import numpy as np
import glob
import os
import argparse



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
opts, _ = parser.parse_known_args()


rt = opts.rt
n_max = opts.n_max
# destination_folder = '/scratch/sehgal.n/datasets/train_surreal_images'
destination_folder = opts.destination_folder
input_folder = '/scratch/sehgal.n/datasets/surreal/SURREAL/data/cmu/train' 

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
                    cv2.imwrite(os.path.join(output_folder, '{}_{}.png'.format(video_name, frame_num)), frame)
                    joints = joint_labels[:, :, frame_num]
                    video_joints.append(joints)

            # Save joints
            output_joints_file = os.path.join(output_folder, '{}.npy'.format(video_name))
            np.save(output_joints_file, video_joints)
            
