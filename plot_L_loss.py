'''
plot the L loss to demonstrate the clip effect.
Others:
'''
import matplotlib
import matplotlib.pyplot as plt
import json
import os.path as osp
import os
import numpy  as np
from utils.utils_tool import make_folder

## set font to bee
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 22,
        }
matplotlib.rc('font', **font)

rst_fd = 'output/GD_PA/result'
pth_La= osp.join(rst_fd, 'ScanAva_Human36M-p2_clip_L.json')
pth_Lb = osp.join(rst_fd, 'ScanAva_Human36M-p2_L.json')

out_fd = 'output/demos'     # for demo figures
if not osp.exists(out_fd):
	os.makedirs(out_fd)

with open(pth_La, 'r') as f:
	dtIn = json.load(f)
	f.close()
arrA = np.array(dtIn)

with open(pth_Lb, 'r') as f:
	dtIn = json.load(f)
	f.close()
arrB = np.array(dtIn)

nm = osp.split(pth_Lb)[1].split('.')[0]     # root nm.ext .split [0]

fig, ax = plt.subplots()
ax.plot(arrA, label='with clip')
ax.plot(arrB, label='without clip')
ax.set_xlabel('Epoch')
ax.set_ylabel('L1 loss')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(osp.join(out_fd, nm + '.pdf'))

# plt.show()




