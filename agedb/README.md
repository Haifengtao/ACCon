# ACCon on AgeDB
This repository contains the implementation of __ConR__ on *AgeDB-DIR* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Gong et al., ICML 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression). the ConR is from [Keramati et al., ConR: Contrastive Regularizer for Deep Imbalanced Regression](https://github.com/BorealisAI/ConR). 



## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data` 

2. Train/val/test split file, which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use these fileS. 
- `agedb_balanced.csv`:  provided by Yang et al.(ICML 2021)
- `agedb_natural.csv`: Randomly splited.
- `agedb_natural_1_2.csv`: half of training data of agedb_natural.



#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget


## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --datatype natural --store_name mse --workers 16 --epoch 90 --regularization_weight 0  --batch_size 64 --lr 0.00025
```



### 2. Train a model with ACCon
##### batch size 64, learning rate 2.5e-4

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model contraNet_ada_emb --datatype balanced --store_name weight_1 --regularization_type comp2 --workers 16 --epoch 100 --proj_dims 512  --temperature 0.05 --regularization_weight 1  --batch_size 64 --lr 0.00025
```



### 3. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the anonymous links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```





