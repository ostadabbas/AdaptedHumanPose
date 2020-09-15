#!/bin/bash
#SBATCH --job-name=PA_GD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/PA_GD.%j.out
#SBATCH --error=bch_outs/PA_GD.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan],testset: Human36M | MuPoTs
#source activate pch1.5
#3=changeInside   # can't work this way
#echo input is ${1} ${2} ${3:-$haha}
# call train_PA_GD for easy modification with sbatch head. Use sh at the moment or uncomment the activate env line.
# mainly work on last line, pelvis,  gt train,  mode 2 skel_av, default ScanAva
# DSA for 50  DSR for 5， if_gtSrc if use gt source or the predicted src , if_gt_PA, if use the test split target skletal descriptor.
cmd=python
$cmd train_PA_GD.py --trainset ${1:-ScanAva} MSCOCO MPII --lmd_D 0.0 --pivot n \
--lmd_G_PA 1 --lmd_D_PA 0. --lr_PA 1e-4 --if_clip_grad y --end_epoch_PA 70 \
--if_ylB y --lmd_skel_PA 50. --gamma_PA 0.95 --if_skel_MSE y --tarset_PA ${2:-h36m-p2} --if_gt_PA n