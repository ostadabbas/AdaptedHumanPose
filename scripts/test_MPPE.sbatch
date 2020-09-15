#!/bin/bash
#SBATCH --job-name=taskGen3d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/taskGen3d.%j.out
#SBATCH --error=bch_outs/taskGen3d.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan]
# single script to run one training session SA 0.02 lsgan ly
# model needs to be changed to  MPPE_h36m-p2 as expNm , test the pretraind performance
source activate pch1.5
python test_MPPE.py \
--model MPPE \
--lmd_D 0.02 \
--mode_D SA \
--pivot sdt \
--gan_mode lsgan \
--if_ylB y \
--testset ${1}  \
--h36mProto ${2} \
--test_par ${3} \
--if_loadPreds n
