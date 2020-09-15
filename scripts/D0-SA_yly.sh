#!/bin/bash
#SBATCH --job-name=D0-SA_yly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/D0-SA_yly.%j.out
#SBATCH --error=bch_outs/D0-SA_yly.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan],testset: Human36M | MuPoTs
source activate pch1.5
python ${1}.py \
--trainset ${2} MSCOCO MPII \
--lmd_D ${3} \
--mode_D SA \
--gan_mode ${4} \
--if_ylB y \
--testset ${5}
