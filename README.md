# Adapted Human Pose (AHuP): 3D Human Pose Estimation with Zero Real 3D Human Pose Data
![poseDemo](imgs/poseDemo.PNG)
## Intro:
AHuP aims at study the effect when a pretrained human pose estimation model is applied under a new environment and the corresponding adaptation approach.  Our working example is from Synthetic human to estimate real human which differes greatly in appearance and also hold "stiff" poses with over simplified skeleton.  Based on this without a single real 3D human data, we achieved comparable performance with the state-of-the-art model based on real human pose data on corresponding benchmarks. SAA can be also employed as a light weighted head to improve the existing SOTA pose model cross set performance with negligible extra computational burden. 

## Env Setup 
The required packages are provided in the requirementes.txt 
Here are some tips for installation. 

For cocoapi for windows: 
* install the [cocoapi](https://github.com/philferriere/cocoapi)
   this version supports windows. For linux, you can also use the offical release.    
install the cocoapi

`pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

 If you don't want to install: just use 
`opts.cocoapi_dir`can be used to point to cocoapi directly 

This repo is developed with python 3.8, CUDA 10.2.  
 
* naming rules:  
SA ScanAva+.  SR, SURREAL  
 
## Data Deployment

*`SURREAL`   data preparation scripts are provided in `dataPrep` folder. First, you need to get a user name and password from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) team first then use our script to generate SURREAL training images, please refer the readme inside `dataPrep`. 

*`ScanAva+`:  Additional collected scans which formed ScanAva+ datasets please check our website for downloading [[link]](https://web.northeastern.edu/ostadabbas/code/)  
  
* For `MSCOCO, MPII, Human3.6M, MuCo, MuPoTS`, please refer to [[link]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE).  
Attention,  for MuPoTS and MuCo, different from the original folder configuration, we remove the middle subfolder `data` and put all contect directly under the dataset name folder.  
 
* setup the dataset dir with `opt.ds_dir`, all datasets should be located under this folder with structure:
（tip: you can use soft link if you do not wanna move existing dataset) 

* SPA training data, download the pose data [data_files](http://www.coe.neu.edu/Research/AClab/AHuP/data_files.zip) and extract under the repo root folder. We also provide the script to extract these pose data if you want to adapt your customized dataset by `gen_poseDs_hm.py`. 

   
```
${ds_dir}
|-- Human36M
|   |-- bbox_root
|   |-- bbox_root_human36m_output.json
|   |-- images
|   `-- annotations
|-- MPII
|   |-- images
|   `-- annotations
|-- MSCOCO
|-- |-- bbox_root
|   |   |-- bbox_root_coco_output.json
|   |-- images
|   |   |-- train/
|   |   |-- val/
|   `-- annotations
|-- MuCo
|  |-- augmented_set
|  |-- unaugmented_set
|  |-- MuCo-3DHP.json
|-- MuPoTS
|   `-- bbox_root
|   |   |-- bbox_mupots_output.json
|   |-- MultiPersonTestSet
|   `-- MuPoTS-3D.json
|-- ScanAva
|   |-- ScanName1
|   |   |-- images
|   |   `-- annotations.pkl
|   |   `-- ... 
| -- train_surreal_images 
|   |-- run0
|   |-- run1
|   |-- run2
|   `-- surreal_annotations_raw.npy
```

## Model deployment
All exps and models are saved in `output`. 
For `ScanAva` demo: 
 
* Semantic Aware Adaptation (SAA) [[link]](http://www.coe.neu.edu/Research/AClab/AHuP/ScanAva-MSCOCO-MPII_res50_D0.02-l3k1-SA-psdt_lsgan_yl-y_regZ5_fG-n_lr0.001_exp.zip) extract to
`output/ScanAva-MSCOCO-MPII_res50_D0.02-l3k1-SA-psdt_lsgan_yl-y_regZ5_fG-n_lr0.001_exp/`
This also includes the initial estimation of SAA  `*pred_hm.json` for later SPA training and test.

* Integral Human (Sun, ICCV'18) trained on Human36M+MSCOCO+MPII [[link]](http://www.coe.neu.edu/Research/AClab/AHuP/Human36M-MSCOCO-MPII_res50_D0.0-l3k1-SA-pn_lsgan_yl-y_regZ5_fG-n_lr0.001_exp.zip). This is to demo the SPA to improve SOTA cross set performance. 

* Hg3d (Zhou, ICCV'17) official Release with our folder structure [[link]](http://www.coe.neu.edu/Research/AClab/AHuP/hg3d.zip). This is to demo the SPA to improve SOTA cross set performance. 

* Skeletal Pose Adaptation (SPA) [[link]](http://www.coe.neu.edu/Research/AClab/AHuP/)
`output/GD_PA
`
The checkpoint files are saved under `model_dump` following the naming rule: `<source_dataset>_<target_dataset>[_best].pth` 


## SAA 
Train the SAA version, ScanAva-SAA-jt2d-SPA , use the default setting  
`python train.py` 
To test on different dataset performance : 
`python test.py --testset [testset_name]` 
Here, testset_name can be [MuPoTS|Human36M|ScanAva|SURREAL] 

## SPA 
SPA which is designed as a light weighted head over existing 3D pose model.
So in our pipeline it is trained after SAA. 
First, generate the initial estimation result of the whole dataset with the SPA model. Then use the initial estimation data to train the SPA model.

### Integral Human (Sun, ICCV'18) with SPA
Here is an example to apply the SPA on integral human (Sun, ICCV'18) trained on 
Human36M+MSCOCO+MPII dataset to improve the its performance over ScanAva+ dataset. 
 
1. Generate the  estimation result for ScanAva training split, 
`. scripts/test_interHum.sbatch Human36M ScanAva train n`
2. Generate the  estimation result for ScanAva test split, 
`. scripts/test_interHum.sbatch Human36M ScanAva test n`
3. Train the SPA on ScanAva 
`. scripts/train_PA_GD.sh Human36M ScanAva` 
4. Evaluate the SPA result 
`. scripts/test_interHum.sbatch Human36M ScanAva test y`

To reproduce the exact result, please use the pretrained model provided in GD_PA.zip extracted under `output/GD_PA`. 
1. generate SPA result. 
`. scripts/test_PA_GD.sbatch`
2. Evaluate the SPA result. 
`. scripts/test_interHum.sbatch Human36M ScanAva test y`

You can follow the same procedure to evaluate SPA over other datasets [ScanAva|SURREAL|MuPoTS] 

### Hg3d (Zhou, ICCV'17) with SPA
We provide the official weights with `hg3d.zip` with file structure for your convenience. You can also download the weights from their official [repo](https://github.com/xingyizhou/pose-hg-3d) and extract under `output/hg3d/model_dump`. 

Generate the original estimation result 
`. scripts/test_hg3d.sbatch ScanAva n`
Generate the SPA result 
`. scripts/test_PA_GD.sbatch Human36M ScanAva hg3d`
Evaluate the SPA result: 
`. scripts/test_hg3d.sbatch ScanAva y`

Here, you can simply get the result for other cross set performance by replacing the `ScanAva` to the dataset you want to test. 

## Citation:
If you find this work helpful, please cite the following papers: 

@article{liu2021adapted,
  title={Adapted Human Pose: Monocular 3D Human Pose Estimation with Zero Real 3D Pose Data},
  author={Liu, Shuangjun and Sehgal, Naveen and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2105.10837},
  year={2021}
}

@inproceedings{liu2018semi,
  title={A semi-supervised data augmentation approach using 3d graphical engines},
  author={Liu, Shuangjun and Ostadabbas, Sarah},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
  pages={0--0},
  year={2018}
}


## Acknowledgement
This repo refers to the following works:
 
https://github.com/mks0601/3DMPPE_POSENET_RELEASE

https://github.com/una-dinosauria/3d-pose-baseline

https://github.com/xingyizhou/pose-hg-3d





