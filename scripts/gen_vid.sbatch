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
# generate video ScanAva, param: 1.testset, 2.iters, run single batch to have more images
source activate pch1.5
python gen_vid.py \
--trainset ScanAva MSCOCO MPII \
--lmd_D 0.02 \
--mode_D SA \
--pivot sdt \
--gan_mode lsgan \
--if_ylB y \
--testset ${1:-Human36M}  \
--test_par demo \
--testIter ${2:-2000} \
--batch_size ${3:-1} \
--svVis_step 1
