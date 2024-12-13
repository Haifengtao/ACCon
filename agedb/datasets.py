# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2021-present, Yuzhe Yang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir
# by Yuzhe Yang et al.
########################################################################################

import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        # 分布的倒数 或者说是 分布开跟的倒数
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel,
                                             lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray(
            [np.float32(1.)])

        return img, label, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def _prepare_weights(self, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


class AgeDB_AdaSCL(data.Dataset):
    def __init__(self, df, data_dir, img_size, n_views=2, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, aug='crop,flip,color,grayscale'):
        self.df = df
        self.aug = aug
        data = df
        data["age_bkt"] = data["age"] // 1  # // 0.02 // 0.01
        EF_freq = data['age_bkt'].value_counts(dropna=False).rename_axis('age_bkt_key').reset_index(name='counts')
        EF_freq = EF_freq.sort_values(by=['age_bkt_key']).reset_index()
        EF_dict = {}
        for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
            if key_itr_idx == 0:
                EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_freq['counts'][key_itr_idx]
            else:
                EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['age_bkt_key'][key_itr_idx - 1]] + \
                                                               EF_freq['age_bkt_key'][key_itr_idx]
        for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
            EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] - \
                                                           EF_freq['age_bkt_key'][key_itr_idx] / 2
        data['age_CLS'] = data["age_bkt"].apply(lambda x: EF_dict[x])
        # import pdb
        # pdb.set_trace()
        self.data_dir = data_dir
        self.img_size = img_size
        self.n_views = n_views
        self.split = split
        # 分布的倒数 或者说是 分布开跟的倒数
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel,
                                             lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform_rnc()

        imgs = []
        for i in range(self.n_views):
            imgs.append(transform(img))

        label = np.asarray([row['age']]).astype('float32')
        label_pdf = np.asarray([row['age_CLS']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray(
            [np.float32(1.)])
        # import pdb
        # pdb.set_trace()
        return imgs, label, label_pdf, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def get_transform_rnc(self):
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if self.split == 'train':
            aug_list = self.aug.split(',')
            transforms_list = []

            if 'crop' in aug_list:
                transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
            else:
                transforms_list.append(transforms.Resize(256))
                transforms_list.append(transforms.CenterCrop(224))

            if 'flip' in aug_list:
                transforms_list.append(transforms.RandomHorizontalFlip())

            if 'color' in aug_list:
                transforms_list.append(transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8))

            if 'grayscale' in aug_list:
                transforms_list.append(transforms.RandomGrayscale(p=0.2))

            transforms_list.append(transforms.ToTensor())
            transforms_list.append(normalize)
            transform = transforms.Compose(transforms_list)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return transform


    def _prepare_weights(self, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import pandas as pd

    print('=====> Preparing data...')
    # print(f"File (.csv): {args.dataset}_{args.datatype}.csv")
    df = pd.read_csv('./data/agedb_balanced.csv')
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    train_dataset = AgeDB(data_dir='./data',  df=df_train, img_size=224,
                                 split='train',
                                 reweight="inverse", lds=True, lds_kernel="gaussian", lds_ks=5,
                                 lds_sigma=2)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=False)

    for i in train_loader:
        import pdb
        pdb.set_trace()
        break
