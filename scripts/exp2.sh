#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -p general
#SBATCH --output=bch_outs/exp2.%j.out
#SBATCH --error=bch_outs/exp2.%j.err

echo called at `date`
sleep 30
