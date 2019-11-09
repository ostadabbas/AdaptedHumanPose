#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -p general
#SBATCH --output=bch_outs/exp1_arg.%j.out
#SBATCH --error=bch_outs/exp1_arg.%j.err
echo run command: ${1}.py --epoch 10