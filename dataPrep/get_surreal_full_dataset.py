'''
Notes
------
Available keys in annotation: dict_keys(['__header__', '__version__', '__globals__', 'sequence', 'clipNo', 'source', 'bg', 'gender', 'light', 'stride', 'camDist', 'camLoc', 'joints2D', 'joints3D', 'pose', 'zrot', 'cloth', 'shape'])
'''
import os
import scipy.io as sio
from tqdm import tqdm
import numpy as np

train_folder = '/scratch/sehgal.n/datasets/train_surreal_images'
original_folder = '/scratch/sehgal.n/datasets/surreal/SURREAL/data/cmu/train'
annotation_file_suffix = '_info.mat'

progress_bar = tqdm(total=211443)
dataset = []

for (dirpath, dirnames, filenames) in os.walk(train_folder):
	for file in filenames:
		if file.endswith('.png'):
			# Get paths
			train_folder_path = dirpath.split('train_surreal_images/')[-1]
			vid_file_name = train_folder_path.split('/')[-1]
			surreal_path = os.path.join(original_folder, train_folder_path.split(vid_file_name)[0])

			# Load annotation
			annotation = sio.loadmat(os.path.join(
				surreal_path,
				vid_file_name + annotation_file_suffix))
			frame_num = int(file.split('.png')[0].split('_')[-1])

			# Get annotation for our frame num
			image_name = os.path.join(train_folder_path, vid_file_name + '_{}.png'.format(frame_num))
			assert os.path.exists(os.path.join(train_folder, image_name)), 'Image doesn\'t exist!'
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
	progress_bar.update()
	dataset.append(this_frame_annotation)

# Save and close out
np.save('surreal_annotations_raw.npy', dataset)
