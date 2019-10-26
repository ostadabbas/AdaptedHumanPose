import os
import os.path as osp
import sys
import numpy as np

# task:
# change all Config thing to parser format, cfg = parser.parse_args() for cluster convenience

class Config:
    
    ## dataset
    # training set
    # 3D: Human36M, MuCo
    # 2D: MSCOCO, MPII 
    # Note that list must consists of one 3D dataset (first element of the list) + several 2D datasets
    trainset = ['Human36M']

    # testing set
    # Human36M, MuPoTS, MSCOCO
    # testset = 'Human36M'
    # testset = 'MSCOCO'
    testset = 'MuPoTS'
    if_normBone = 'n'
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_tool_dir = osp.join(root_dir, 'data')
    ds_dir = '/scratch/liu.shu/datasets'        # discovery true locations
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    cocoapi_dir = '/scratch/liu.shu/codesPool/cocoapi/PythonAPI'        # cluster coco
    ## model setting
    resnet_type = 50 # 50, 101, 152
    
    ## input, output
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)  # perhaps for RGB normalization  after divide by 255
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    lr_dec_epoch = [17, 21]
    end_epoch = 2   # default 25,  2 for test purpose
    lr = 1e-3
    lr_dec_factor = 10
    batch_size = 32

    ## testing config
    test_batch_size = 16
    flip_test = True
    use_gt_info = False

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, cfg.root_dir)
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.utils_tool import add_pypath, make_folder
add_pypath(osp.join(cfg.data_tool_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_tool_dir, cfg.trainset[i]))
add_pypath(cfg.cocoapi_dir)     # add coco dir to it
add_pypath(osp.join(cfg.data_tool_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

