# Imbalanced Visual Rcongnition by PPL with CRI Loss

Our code is based on [MisLAS](https://arxiv.org/pdf/2104.00466.pdf) and [RIDE](https://people.eecs.berkeley.edu/~xdwang/papers/RIDE.pdf) models.

## Installation

### Requirements

* numpy==1.22.0  
* python==3.9
* pytorch==1.10.1
* torchvision==0.11.2
* tqdm==4.62.3
* pillow==8.4.0

### Dataset Preparation

* [cifar10 & cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

* [ImageNet](http://image-net.org/index)

* [iNaturalist2018](https://github.com/visipedia/inat_comp/tree/master/2018)

[data_txt file Link](https://drive.google.com/drive/folders/1ssoFLGNB_TM-j4VNYtgx9lxfqvACz-8V?usp=sharing)

For origin_PPL+CRI, change the `data_path` in `config/.../.yaml`

For PPL+CRI multi experts:
  
1) change the `class ImbalanceCIFAR10DataLoader/data_dir` or `class ImbalanceCIFAR100DataLoader/data_dir` in `./data_loader/cifar_data_loaders.py`;
2) change the `data_dir` and `txt_train_dir` in `./data_loader/imagenet_lt_loaders.py`;
3) change the `data_dir` and `txt_train_dir` in `./data_loader/inaturalist_data_loaders.py`.

## Training

one GPU for Imbalance cifar10 & cifar100, two GPUs for ImageNet-LT, and eight GPUs iNaturalist2018.

Backbone network can be resnet32 for Imbalance cifar10 & cifar100, resnet10 for ImageNet-LT, and resnet50 for iNaturalist2018.

### origin_PPL+CRI

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10/cifar10_CRI.yaml`

`python train.py --cfg ./config/cifar100/cifar100_CRI.yaml`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet/imagenet_CRI.yaml`

#### ina2018:

`python train.py --cfg ./config/ina2018/ina2018_CRI.yaml`

### PPL+CRI multi experts

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10.json`

`python train.py --cfg ./config/cifar100.json`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet.json`

#### ina2018:

`python train.py --cfg ./config/ina2018.json`

## Results and Models

### origin_PPL+CRI  

[Links to models](https://drive.google.com/drive/folders/1b932TjGm_-GcuN9Mq24aExk2uZK64LWy?usp=sharing)

### PPL+CRI multi experts

[Links to models](https://drive.google.com/drive/folders/1Dqh0Jcs-lqKv0BkEJmMX8JJwnhCL7mhx?usp=sharing)

## More content to be updated



