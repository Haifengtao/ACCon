# ACCON on IMDB-WIKI
This repository contains the implementation of __ACCon__ on *IMDB-WIKI* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Gong et al., ICML 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression). the ConR is from [Keramati et al., ConR: Contrastive Regularizer for Deep Imbalanced Regression](https://github.com/BorealisAI/ConR). 



## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. Train/val/test split file, which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use these fileS. 
- `imdb_wiki_balanced.csv`:  provided by Yang et al.(ICML 2021)
- `imdb_wiki_natural.csv`: Randomly splited.


#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget


## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
python train.py --batch_size 64 --lr 2.5e-4
```



### 2. Train a model with ACCon
##### batch size 64, learning rate 2.5e-4

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --model contraNet_ada_emb --datatype balanced --store_name weight_1 --regularization_type accon --workers 24 --epoch 90 --proj_dims 512  --temperature 0.05 --regularization_weight 1  --batch_size 64 --lr 0.00025
```



### 3. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the anonymous links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```




