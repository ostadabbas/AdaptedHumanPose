#!/bin/bash
#SBATCH --job-name=SA_A
#SBATCH --mem=30Gb
#SBATCH --output=SA_A.%j.out
#SBATCH --error=SA_A.%j.err

#python3 sne_plot.py -f /scratch/liu.shu/codesPool/AHuP/output_cvpr20/ScanAva-MSCOCO-MPII_res50_n-scraG_10.0D2_y-yl_1rtSYN_regZ0_n-fG_n-nmBone_adam_lr0.001_exp/vis/train -o sa_aaic_200.png -m 200
python3 sne_plot.py -f /scratch/liu.shu/codesPool/AHuP/output_cvpr20/ScanAva-MSCOCO-MPII_res50_n-scraG_0.1D2_y-yl_1rtSYN_regZ10_n-fG_y-nmBone_adam_lr0.001_exp/vis/train -o sa_aaic_200.png -m 200

