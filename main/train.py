import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator(ds_dir=cfg.ds_dir)
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr in range(trainer.itr_per_epoch):
            
            input_img_list, joint_img_list, joint_vis_list, joints_have_depth_list = [], [], [], []
            for i in range(len(cfg.trainset)):  # loop set
                try:
                    input_img, joint_img, joint_vis, joints_have_depth = next(trainer.iterator[i])      # iterator, bch_gen all list. next bch
                except StopIteration:
                    trainer.iterator[i] = iter(trainer.batch_generator[i])  # set again suppose to be set already
                    input_img, joint_img, joint_vis, joints_have_depth = next(trainer.iterator[i])

                input_img_list.append(input_img)
                joint_img_list.append(joint_img)
                joint_vis_list.append(joint_vis)
                joints_have_depth_list.append(joints_have_depth)
            
            # aggregate items from different datasets into one single batch
            input_img = torch.cat(input_img_list,dim=0)
            joint_img = torch.cat(joint_img_list,dim=0)
            joint_vis = torch.cat(joint_vis_list,dim=0)
            joints_have_depth = torch.cat(joints_have_depth_list,dim=0)
            
            # shuffle items from different datasets
            rand_idx = []
            for i in range(len(cfg.trainset)):
                rand_idx.append(torch.arange(i,input_img.shape[0],len(cfg.trainset)))   # len(trainSet) interval 0,2,4,....then [1,3,5...]  not necessary batch is not input channel actually
            rand_idx = torch.cat(rand_idx,dim=0)
            rand_idx = rand_idx[torch.randperm(input_img.shape[0])]
            input_img = input_img[rand_idx]; joint_img = joint_img[rand_idx]; joint_vis = joint_vis[rand_idx]; joints_have_depth = joints_have_depth[rand_idx];
            target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth}

            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()
            
            # forward
            loss_coord = trainer.model(input_img, target) # direct loss?
            loss_coord = loss_coord.mean()

            # backward
            loss = loss_coord

            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()

            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_coord', loss_coord.detach()),
                ]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save every epoch?
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        

if __name__ == "__main__":
    main()
