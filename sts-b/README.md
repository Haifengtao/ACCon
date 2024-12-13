

# ACCON ON STS-B
This repository contains the implementation of __ACCon__ on *STS-B* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Gong et al., ICML 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression). the ConR is from [Keramati et al., ConR: Contrastive Regularizer for Deep Imbalanced Regression](https://github.com/BorealisAI/ConR). 



## Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. __(Optional)__ We have provided both original STS-B dataset and our created balanced STS-B-DIR dataset in folder `./glue_data/STS-B`. To reproduce the results in the paper, please use our created STS-B-DIR dataset. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```

#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.1 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
# The current latest "overrides" dependency installed along with allennlp 0.5.0 will now raise error. 
# We need to downgrade "overrides" version to 3.1.0
pip install overrides==3.1.0
```

## Getting Started

#### Train a vanilla model

```bash
CUDA_VISIBLE_DEVICES=0 python train_mine.py --datatype natural --store_name vanilla --temperature 0.05 --patience 30 --regularization_weight 10  --batch_size 128 --lr 1e-4 --loss mse
```


#### Train a model using ACCON

```bash
CUDA_VISIBLE_DEVICES=0 python train_mine.py --datatype natural --store_name ACCon --regularization_type comp2 --proj_dims 2000 --temperature 0.05 --patience 30 --regularization_weight 10  --batch_size 128 --lr 1e-4 --loss mse
```

#### Evaluate a trained checkpoint

```bash
python train.py [...evaluation model arguments...] --evaluate --eval_model <path_to_evaluation_ckpt>
```
