import os
import numpy as np
import torch
import random
from glob import glob
import itertools
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader, _SingleProcessDataLoaderIter
from torch.utils.data.sampler import Sampler, SequentialSampler, \
    RandomSampler
from torchvision import transforms
from PIL import Image
from scipy import io, misc


class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir, embedding_otf=False,
                 transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = [x.split('\\')[-1] for x in glob(cover_dir + '/*')]
        self.transform = transform
        self.embedding_otf = embedding_otf
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        paths = []
        labels = np.array([0, 1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        cover = Image.open(cover_path)
        images = np.empty((2, cover.size[0], cover.size[1], 1),
                          dtype='uint8')
        images[0, :, :, 0] = np.array(cover)
        paths.append(cover_path)
        if self.embedding_otf:
            images[1, :, :, 0] = np.copy(images[0, :, :, 0])
            beta_path = os.path.join(self.stego_dir,
                                     '.'.join(self.cover_list[idx].
                                              split('.')[:-1]) + '.mat')
            beta_map = io.loadmat(beta_path)['pChange']
            rand_arr = np.random.rand(cover.size[0], cover.size[1])
            inf_map = rand_arr < (beta_map / 2.)
            images[1, np.logical_and(
                images[0, :, :, 0] != 255, inf_map), 0] += 1
            inf_map[:, :] = rand_arr > 1 - (beta_map / 2.)
            images[1, np.logical_and(images[0, :, :, 0] != 0, inf_map), 0] -= 1
        else:
            stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
            images[1, :, :, 0] = Image.open(stego_path)
            paths.append(stego_path)
        samples = {'images': images, 'labels': labels, "paths": paths}
        if self.transform:
            samples = self.transform(samples)
        return samples


class RandomBalancedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        cover_perm = [x * 2 for x in torch.randperm(
            len(self.data_source) / 2).long()]
        stego_perm = [x * 2 + 1 for x in torch.randperm(
                      len(self.data_source) / 2).long()]
        return iter(it.next() for it in
                    itertools.cycle([iter(cover_perm), iter(stego_perm)]))

    def __len__(self):
        return len(self.data_source)


class DataLoaderIterWithReshape(_SingleProcessDataLoaderIter):
    def next(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self._reshape(self.collate_fn(
                                  [self.dataset[i] for i in indices]))
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._reshape(self._process_next_batch(batch))

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._reshape(self._process_next_batch(batch))

    def _reshape(self, batch):
        images, labels = batch['images'], batch['labels']
        shape = list(images.size())
        return {'images': images.view(shape[0] * shape[1], *shape[2:]),
                'labels': labels.view(-1)}


class DataLoaderStego(DataLoader):
    def __init__(self, cover_dir, stego_dir, embedding_otf=False,
                 shuffle=False, pair_constraint=False, batch_size=1,
                 transform=None, num_workers=0, pin_memory=False):
        self.pair_constraint = pair_constraint
        self.embedding_otf = embedding_otf
        if pair_constraint and batch_size % 2 == 0:
            dataset = DatasetPair(cover_dir, stego_dir, embedding_otf,
                                  transform)
            _batch_size = batch_size / 2
        else:
            print("nopair")
            dataset = DatasetNoPair(cover_dir, stego_dir, embedding_otf,
                                    transform)
            _batch_size = batch_size
        if pair_constraint:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        else:
            sampler = RandomBalancedSampler(dataset)

        super(DataLoaderStego, self). \
            __init__(dataset, int(_batch_size), None, sampler,
                     None, num_workers, pin_memory=pin_memory, drop_last=True)
        self.shuffle = shuffle

    def __iter__(self):
        return DataLoaderIterWithReshape(self)


class ToTensor(object):
    def __call__(self, samples):
        images, labels, paths = samples['images'], samples['labels'], samples['paths']
        images = images.transpose((0, 3, 1, 2))
        images = torch.from_numpy(images)
        images = images.type(torch.FloatTensor)
        return {'images': images,
                'labels': torch.from_numpy(labels).long(), 'paths': paths}

