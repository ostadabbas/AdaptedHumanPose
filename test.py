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
import os.path as osp
from utils.logger import Colorlogger
from models.modelTG import TaskGenNet
from data.dataset import genLoaderFromDs
import utils.utils_tool as ut_t
import math

def testLoop(model, ds_test, logger_test='test_tmp.txt', if_svEval=False, if_svVis=False):
    '''
    generate loader from ds, then test the model, logger record, and image save directly here.
    :param model: the trained model
    :param ds_test: the test dataset
    :param logger_test: to record continuously
    :return:
    '''
    # ds, loader, iterator = dataset.genDsLoader(opts, mode=mode)


    ds_adp, loader, iter_test = genLoaderFromDs(ds_test, opts)   # will be in batch format
    preds = []

    itr_per_epoch = math.ceil(
        ds_adp.__len__() / opts.num_gpus / opts.batch_size)  # single ds test on batch share
    if opts.testIter > 0:
        itr_per_epoch = min(itr_per_epoch, opts.testIter)

    # for i, (input, target) in enumerate(loader):
    for i in range(itr_per_epoch):
        input, target = next(iter_test)
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

        if if_svVis and 0 == i % opts.svVis_step:
            # save visuals  only 1st one in batch
            sv_dir = opts.vis_test_dir      # exp/vis/Human36M
            img_patch_vis = ut_t.ts2cv2(input['img_patch'][0])
            idx_test = i* opts.batch_size
            skel_idx = opts.ref_skels_idx
            # get pred2d_patch
            pred2d_patch = np.zeros((3, opts.joint_num))        # 3xn_jt format
            pred2d_patch[:2, :] = coord_out[0, :, :2].cpu().numpy().transpose(1, 0) / opts.output_shape[0] * opts.input_shape[0]  # x * n_jt ?
            pred2d_patch[2, :] = 1
            ut_t.save_2d_tg3d(img_patch_vis, pred2d_patch, skel_idx, sv_dir, idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
            ut_t.save_hm_tg3d(HM.cpu().numpy(), sv_dir, idx=idx_test)        # only save the first HM here,
            ut_t.save_Gfts_tg3d(G_fts.cpu().numpy(), sv_dir, idx=idx_test)   # only first 25 hm, also histogram of G_fts
            # HM front and side view
            ut_t.save_3d_tg3d(coord_out, sv_dir, skel_idx, idx=idx_test, suffix='hm')       # if need mm plot, can be done in eval part with ds infor, here only for HM version

    preds = np.concatenate(preds, axis=0)  # x,y,z :HM preds = np.concatenate(preds, axis=0)   # x,y,z :HM  into vertical long one
    err_dict = ds_test.evaluate(preds, jt_adj=opts.adj, if_svVis=if_svVis, if_svEval=if_svEval)    # shoulddn't return, as different set different metric, some only save
    return err_dict



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

    ds_test = eval(opts.testset)("test", opts=opts)  # keep a test set
    model = TaskGenNet(opts)  # with initialization already, already GPU-par
    if 0 == opts.start_epoch and 'y' == opts.if_scraG:
        model.load_bb_pretrain()  # init backbone
    elif opts.start_epoch > 0:  # load the epoch model
        model.load_networks(opts.start_epoch - 1)
    err_dict = testLoop(model, ds_test, logger_test=logger_test, if_svEval=True, if_svVis=True)



if __name__ == "__main__":
    main()
