#!/bin/bash
#SBATCH --job-name=gen_gtHm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short   # multigpu
##SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/gen_gtHm.%j.out
#SBATCH --error=bch_outs/gen_gtHm.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan]
# generate pose dataset in hm
source activate pch1.5
python gen_poseDs_hm.py --dsNm ${1}
