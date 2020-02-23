import numpy as np
from sklearn.neighbors import NearestNeighbors

x = np.load('all_joints.npy')
joints = x[:, 0]
joints = np.array(list(joints))

new_shape = (len(joints), np.prod(joints.shape[1:]).astype(int))
joints = np.reshape(joints, new_shape)

model = NearestNeighbors()
distances, indicies = model.kneighbors(joints)

