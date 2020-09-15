#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -J mainJobSubmitter
#SBATCH -p short

# call job one by one
#change paths to your desired locations:
SOURCEPATH=scripts  # call this from a sub folder of main
synset=ScanAva    # the synthetic synset
testset=Human36M
ganMode=lsgan
#cmd=echo
cmd=sbatch
lmd_D=0.02
test_par=test
# input
# problem node d1005, 1010, 1020;  1016 possible mem issue
# 1 set, 2 synset, 3 lmd_D, 4 mode_D [C|SA], 5 pivot, 6 gan_mode lsgan|vanilla, 7 if_ylB, 8 testset
#sbatch --exclude=d1005,d1010,d1020 ${SOURCEPATH}/single.sbatch ${2} ${2} ${3} ${4} ${5} ${ganMode} ${7} ${testset}

# real
#$cmd --exclude=d1005,d1010,d1020 ${SOURCEPATH}/single.sbatch ${1} Human36M 0.0 SA n ${ganMode} y ${testset}
#$cmd --exclude=d1005,d1010,d1020 ${SOURCEPATH}/single.sbatch ${1} MuCo 0.0 SA n ${ganMode} y ${testset}

# lmd_D 0.0
#$cmd --exclude=d1005,d1010,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva 0.0 SA n ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva 0.0 SA n ${ganMode} y ${testset}

# scanAva loop for D, 1 train 2 lmd_D
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} C n ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} C uda ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} C sdt ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} C sdt ${ganMode} y ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} SA n ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} SA uda ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} SA sdt ${ganMode} n ${3:-$testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ${synset} ${2:-$lmd_D} SA sdt ${ganMode} y ${3:-$testset}

# test with name control  for SA 0.02 lsagan ly
# test_sgl 1: synset 2: testset 3: proto 4 test_par, test_par train,  only SA, sdt,  3 test set
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL Human36M 1 train
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL Human36M 2 train
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL MuCo 2 train
# ran the 5 ts set with 3 more settings on  ScanAva   exp: temp  only get h36m 2 id-4
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} ScanAva 2 ${test_par}    #
#sleep 10    # wait to create the ${test_par} fd
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} SURREAL 2 ${test_par}
$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} Human36M 1 ${test_par}
$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} Human36M 2 ${test_par}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} MuCo 2 ${test_par}
$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl${2}.sbatch ${1} MuPoTS 2 ${test_par}


# test_sgl 1: synset 2: testset 3: proto 4 test_par test_par test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL Human36M 1 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL Human36M 2 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch SURREAL MuPoTS 2 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch ScanAva Human36M 1 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch ScanAva Human36M 2 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_sgl1.sbatch ScanAva MuPoTS 2 test

# test_MEEP 1: synset 2: testset 3: proto 4 test_par test_par test, only h36m  the other has joint issue.
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch Human36M 1 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch Human36M 2 test
##$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch MuPoTS 2 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch Human36M 1 train
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch Human36M 2 train
##$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/test_MPPE.sbatch MuCo 2 train

# C1 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} C1 n ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} C1 uda ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} C1 sdt ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} C1 sdt ${ganMode} y ${testset}

# SA1 test
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} SA1 n ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} SA1 uda ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} SA1 sdt ${ganMode} n ${testset}
#$cmd --exclude=d1005,d1010,d1016,d1020 ${SOURCEPATH}/single.sbatch ${1} ScanAva ${2} SA1 sdt ${ganMode} y ${testset}