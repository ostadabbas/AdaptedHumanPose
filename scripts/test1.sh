#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/test1.%j.out
#SBATCH --error=bch_outs/test1.%j.err
# input train/test, dsSyn, gan_mode[vanilla |lsgan],testset: Human36M | MuPoTs
#source activate pch1.5
#python ex_basics.py
#python matplot3d_test.py
#3=changeInside   # can't work this way
haha=newCmd
echo input is ${1} ${2} ${3:-$haha}