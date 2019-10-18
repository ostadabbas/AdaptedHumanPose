'''
A tester function to test model against given loader and
'''
import argparse
from tqdm import tqdm
import numpy as np
import cv2
# from config import cfg
import torch
# from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
from data import dataset
import torch.backends.cudnn as cudnn
from opt import opts, print_options
import os


def testLoop(model, mode=2, if_svRst=False, if_svVis=False):
    '''
    generate loader from ds, then test the model
    :param model: the trained model
    :param mode: the mode for loader generation
    :return:
    '''
    ds, loader, iterator = dataset.genDsLoader(opts, mode=mode)
    preds = []
    for i, data in enumerate(loader):
        model.set_input(data)
        model.forward()
        # get output and G_fts(back_bone feature for future test purpose)
        # preds append
        if if_svVis and 0 == i % opts.svVis_step:        # save G_fts to demo
            print('not implemented yet'); pass # todo ---
            # save G_fts

    rst = ds.evaluate(preds, result_dir=opts.result_dir, if_svRst=if_svRst, if_svVis=if_svVis)    # rst in dict with MPJPE, PCKh, and AUC
    return rst



def main():
    gpu_ids_str = [str(i) for i in opts.gpu_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)  #
    print('>>> Using GPU: {}'.format(gpu_ids_str))
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False         # result unchanged
    cudnn.enabled = True
    print('---test begins---')
    print_options(opts)     # show options
    # test Logger here

    # create model, load in the model we need with epoch number specified in opts
    model = None        # todo ---
    rst = testLoop(model, mode=1, if_svRst=True, if_svVis=True)   # formal test with all

    # form the final test result
	# print
	# log it


if __name__ == "__main__":
    main()
