## Installation

[I](https://github.com/think1920) have tested the code with `python==3.10.10` and `pytorch==1.12.1`, other late versions may also work well. 
<br>
Welcome to provide feedback or suggestion for the version list!
<!-- Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch dependencies. 
Installing PyTorch and TorchVision with CUDA support is strongly recommended. -->

Install FoodSAM with the following steps:

a. Clone the repository locally:
```
https://github.com/think1920/FoodSAM.git
```
b. Create a conda virtual environment and activate it (Optional)
```
conda create -n FoodSAM python=3.10.10 -y
conda activate FoodSAM
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here I use PyTorch 1.12.1 and CUDA 11.3. You may also switch to another version by specifying the version number.
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
d. Install MMCV following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation). 
```
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
e. Install SAM following official [SAM installation](https://github.com/facebookresearch/segment-anything).
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
f. other requirements
```
pip install -r requirement.txt
```
e. Finally download the checkpoints folder, and move it to main project.

[SAM-vit-h and FoodSeg103-SETR-MLA and UNIDET-Unified_learned_OCIM_RS200_6x+2x](https://drive.google.com/drive/folders/1uYRKuY-c_ebB9rx_lc6axp_XvqlIGfKa?usp=sharing)


## Dataset and configs
For UNIDET and FoodSeg103, the configs are already put into the [configs](configs/) folder. 
You can also download other ckpt and configs from their official links.

The default dataset [we](https://github.com/jamesjg) use is [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), other semantic segmentation food datasets like [UECFOODPIXCOMPLETE](https://mm.cs.uec.ac.jp/uecfoodpix/) can also be used. But you should change the  `args.category_txt and args.num_class`. The dataset should be put in the "dataset/"folder.

Your data, configs, and ckpt path should look like this:
````
FoodSAM
-- ckpts
   |-- SETR_MLA
   |   |-- iter_80000.pth
   |-- sam_vit_h_4b8939.pth
   |-- Unified_learned_OCIM_RS200_6x+2x.pth
-- configs
   |-- Base-CRCNN-COCO.yaml
   |-- Unified_learned_OCIM_RS200_6x+2x.yaml
   |-- SETR_MLA_768x768_80k_base.py
-- dataset
   |-- FoodSeg103
   |   |-- Images
   |   |   |-- ann_dir
   |   |   |-- img_dir  
-- FoodSAM
-- mmseg
-- UNIDET
   ...

````


