#!/bin/bash
#SBATCH --job-name=testh36m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/testh36m.%j.out
#SBATCH --error=bch_outs/testh36m.%j.err
# test the real 3d based models
source activate pch1.5
python test.py \
--trainset ${1:-Human36M} MSCOCO MPII \
--lmd_D 0. \
--mode_D SA \
--pivot n \
--gan_mode lsgan \
--if_ylB y \
--testset ${2} \
--test_par ${3:-test} \
--if_loadPreds ${4:-y} \
--if_hm_PA y

python collectRst.py \
--synset ${1:-Human36M} \
--testset ${2} \
--lmd_D 0.
