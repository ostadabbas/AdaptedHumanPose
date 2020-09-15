#!/bin/bash
#SBATCH --job-name=D002-SA-pn_yln
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/D002-SA-pn_yln.%j.out
#SBATCH --error=bch_outs/D002-SA-pn_yln.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan]
source activate pch1.5
python ${1}.py \
--trainset ${2} MSCOCO MPII \
--lmd_D 1. \
--mode_D SA \
--pivot n \
--gan_mode ${3} \
--if_ylB n \
--testset ${4}

