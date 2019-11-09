#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -J mainJobSubmitter
#SBATCH -p general

# change exp names to experiment needs, each run a python train/test codes
#change paths to your desired locations:
SOURCEPATH=scripts_v2  # call this from a sub folder of main
#CURRDIR=/scratch/username/...
N=3
declare -a arr_exp=(
"D0_R2"
#"D0_z10_nfG_ylB_yNB_nScra"
#"D01_z10_nfG_ylB_yNB_nScra"
##"D001_z0_nfG_ylB_yNB_nScra"
#"D001_z0_nfG_ylB_yNB_nScra_nNB"
#"D001_z2_scra_3-10-20"
#"D001_z10_nfG_nlB_yNB_nScra"
#"D001_z10_nfG_ylB_yNB_nScra"
#"D001_z10_nfG_ylB_yNB_yScra"
"D001_z10_yfG_ylB_yNB_nScra"
                )

for i in "${!arr_exp[@]}"; do
    for j in $(seq 1 $N)
    do
#    if [ "$j" -eq "1" ]; then
#    JOBID=`sbatch ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`  # we can easily see it by same name + jobID
#    else
#    JOBID=`sbatch --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`
#    fi
    JOBID=-1
    echo main sbatch job $i ${SOURCEPATH}/${arr_exp[$i]}.sh $1 sub $j with ID $JOBID at `date`
    sleep 1
#    rst=`echo prepare for job $i ${SOURCEPATH}/${arr_exp[$i]}.sh as mode $1`
#    echo rst is $rst
    done
done
#sbatch ${SOURCEPATH}/exp1_arg.sh $1