from pathlib import Path
import numpy as np
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler


class SSLDataLoader(object):
    def __init__(self, labeled_dset, unlabeled_dset, bsl, bsu, num_workers):
        bs = bsl + bsu # 64 + 128
        sampler_lab = InfBatchSampler(len(labeled_dset), bsl)
        sampler_unlab = InfBatchSampler(len(unlabeled_dset), bsu)
        self.labeled_dset = DataLoader(labeled_dset, batch_sampler=sampler_lab, num_workers=int(num_workers*bsl/bs))
        self.unlabeled_dset = DataLoader(unlabeled_dset, batch_sampler=sampler_unlab, num_workers=int(num_workers*bsu/bs))

        self.labeled_iter = iter(self.labeled_dset)
        self.unlabeled_iter = iter(self.unlabeled_dset)

    def __iter__(self):
        return self

    def __next__(self): # label이 있는 데이터의 다음 배치를 가져오고 그 다음으로 레이블이 없는 데이터의 다음 배치를 가져온다.
        try:
            xl, yl = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_dset)
            xl, yl = next(self.labeled_iter)

        try:
            xu = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_dset)
            xu = next(self.unlabeled_iter)

        return xl, yl, xu # xl : (64, 3, 32, 32), yl:(64), xu : (128, 3, 32, 32)


class InfBatchSampler(Sampler):
    def __init__(self, N, batch_size):
        self.N = N
        self.batch_size = batch_size if batch_size < N else N
        self.L = N // batch_size

    def __iter__(self):
        while True:
            idx = np.random.permutation(self.N)
            for i in range(self.L):
                yield idx[i*self.batch_size:(i+1)*self.batch_size]

    def __len__(self):
        return sys.maxsize


class SSLDataset(Dataset):
    def __init__(self, x, y, Taggr, Tsimp, K, shape):
        super().__init__()

        self.x = x # (250, 32, 32, 3)
        self.y = y # (250, )
        self.Taggr = Taggr # augmentation, Aggressive 
        self.Tsimp = Tsimp # augmentation, Simple
        self.K = K # None
        self.shape = shape # 32

    def read_x(self, idx):
        raise NotImplementedError

    def get_x(self):
        x = []
        for idx in range(len(self.x)):
            xi = self.read_x(idx)
            x.append(xi)
        return x

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        """
        static method to split data into train/val/test and lab/unlab sets
        :param root_dir: pathlib.Path object. root dir of the dataset.
        :param tgt_domains: list of str. list of target domains.
        :param src_domains: list of str. list of source domains.
        :param r_val: number. ratio of validation set from the train set.
        :param r_lab: number. ratio of labeled data from the target train set.
        :param r_unlab: number. ratio of unlabeled data between the source train set and the target train set.
        :param w_unlab: list of numbers. sampling weights for unlabeled source sets.
        :param rand_seed: number. random seed.
        :param r_data: number. ratio of data to consider.
        :return xl, yl, xu, xv, yv, xt, yt: different data splits.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xi = self.read_x(idx)

        if self.K == 0:
            x = self.Taggr(xi)
            x = x.squeeze(0)
        elif self.K is not None:
            x = [self.Tsimp(xi)]
            for _ in range(self.K):
                x.append(self.Taggr(xi))
            x = torch.stack(x) # 차원이 하나 더 늘어남, (9, 3, 32, 32)
        else:
        # simple aug 또는 aggressive aug 중에 하나만 하자
            x = self.Tsimp(xi)
        if self.y is not None:
            return x, self.y[idx]
        else:
            return x


class SupDataset(Dataset):
    def __init__(self, x, y, T, shape):
        super().__init__()
        self.x = x
        self.y = y
        self.T = T
        self.shape = shape

    def read_x(self, idx):
        raise NotImplementedError

    @staticmethod
    def split_data(root_dir, domain, r_val, r_data, rand_seed):
        """
        static method to split data into train/val/test and lab/unlab sets
        :param root_dir: pathlib.Path object. root dir of the dataset.
        :param domain: str. target domain.
        :param r_val: number. ratio of validation set from the train set.
        :param r_data: number. ratio of data to consider.
        :param rand_seed: number. random seed.
        :return x, y, xv, yv, xt, yt: different data splits.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.T(self.read_x(idx)), self.y[idx]
