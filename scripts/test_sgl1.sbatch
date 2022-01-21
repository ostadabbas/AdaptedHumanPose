#!/bin/bash
#SBATCH --job-name=AHuP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/AHuP.%j.out
#SBATCH --error=bch_outs/AHuP.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan]
# single script to run one training session SA 0.02 lsgan ly
# 4 test on ScanAva   0.0  SA, pivot n  yLB n
#  0.02  C,  sdt , n
#  0.02  C,  sdt , y
#  0.02 SA, sdt  ylB n
#  0.02 SA, sdt ,  ylB y  done
source activate pch1.5
python test.py \
--trainset ${1} MSCOCO MPII \
--lmd_D 0.0 \
--mode_D SA \
--pivot n \
--gan_mode lsgan \
--if_ylB n \
--testset ${2}  \
--h36mProto ${3} \
--test_par ${4} \
--if_loadPreds n
