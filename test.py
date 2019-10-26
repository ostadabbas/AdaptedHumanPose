'''
A tester function to test model against given ds.
Give the start_epoch to be model_epoch+1, or just give 1 for latest
'''
import argparse
from tqdm import tqdm
import numpy as np
import cv2
# from config import cfg
import torch
# from base import Tester
from utils.vis import vis_keypoints
from utils.utils_pose import flip
from data import dataset
import torch.backends.cudnn as cudnn
from opt import opts, print_options
import os
from utils.logger import Colorlogger
from models.modelTG import TaskGenNet
from data.dataset import genLoaderFromDs


def testLoop(model, ds_test, logger_test='test_tmp.txt', if_svRst=False, if_svVis=False):
    '''
    generate loader from ds, then test the model, logger record, and image save directly here.
    :param model: the trained model
    :param ds_test: the test dataset
    :param logger_test: to record continuously
    :return:
    '''
    # ds, loader, iterator = dataset.genDsLoader(opts, mode=mode)


    ds_adp, loader, iterator = genLoaderFromDs(ds_test, opts)
    preds = []

    for i, (input, target) in enumerate(loader):
        model.set_input(input, target)
        model.forward()
        coord_out = model.coord
        HM = model.HM.cpu().numpy()       # for plotting  purpose later   clone to detach
        G_fts = model.G_fts.cpu().numpy()  # to cpu  mem
        if opts.flip_test:
            img_patch = input['img_patch']
            flipped_input_img = flip(img_patch, dims=3)
            model.set_input({'img_patch': flipped_input_img})
            model.forward()
            flipped_coord_out = model(flipped_input_img)
            flipped_coord_out[:, :, 0] = opts.output_shape[1] - flipped_coord_out[:, :, 0] - 1      # flip x coordinates
            for pair in opts.ref_flip_pairs:
                flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1], :].clone(), flipped_coord_out[:, pair[0], :].clone()
            coord_out = (coord_out + flipped_coord_out) / 2.
        preds.append(coord_out) # add result

        if if_svVis and 0 == i % opts.svVis_step:        # save G_fts and HM, img patch, plot skeleton with coord is good enough.
            print('sv img not implemented yet'); pass # todo ---
            # save G_fts
            # save HM fron_view and side_view hm.


    rst = ds_test.evaluate(preds, result_dir=opts.result_dir, adj=opts.adj, if_svRst=if_svRst, if_svVis=if_svVis)    # rst in dict with MPJPE, PCKh, and AUC
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
    logger_test = Colorlogger(opts.log_dir, 'test_logs.txt')
    # create model, load in the model we need with epoch number specified in opts
    model = TaskGenNet(opts)  # with initialization already, already GPU-par
    if 0 == opts.start_epoch and 'y' == opts.if_scraG:
        model.load_bb_pretrain()  # init backbone
    elif opts.start_epoch > 0:  # load the epoch model
        model.load_networks(opts.start_epoch - 1)
    rst = testLoop(model, mode=1, if_svRst=True, if_svVis=True)   # formal test with all

    # form the final test result
	# print
	# log it


if __name__ == "__main__":
    main()
