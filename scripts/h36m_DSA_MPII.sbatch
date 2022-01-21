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
# single script to run one training session
#  train ScanAva  0.02  SA n Human36M.    pvt can be sdt
# pvt=n
source activate pch1.5
python ${1:-train}.py \
--trainset Human36M ScanAva MPII \
--lmd_D 0.02 \
--mode_D SA \
--pivot sdt \
--gan_mode lsgan \
--if_ylB y \
--testset ${2:-Human36M}


