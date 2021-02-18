#! /usr/bin/env python3.8
from . import config
from torch import argmax
from random import Random
from math import ceil
from torchvision import datasets, transforms

class Partition(object):
  def __init__(self, data, idx):
    self.data = data
    self.idx = idx

  def __len__(self):
    return len(self.idx)

  def __getitem__(self, i):
    return self.data[self.idx[i]]

def gen_partitions(data, indexes, sizes):
  dataset = []
  for i in range(0, len(sizes)):
    size = int(sizes[i]*len(indexes))
    indexes, part_idx = indexes[size:], indexes[:size]
    partition = Partition(data, part_idx)
    dataset.append(partition)
  return dataset

def gen_iid_dataset(data, seed = config.rand_seed, sample_num = config.sample_num, sizes = config.data_distribution):
  engine = Random()
  engine.seed(seed)
  idx = engine.sample(range(len(data)), sample_num)
  engine.shuffle(idx)
  return gen_partitions(data, idx, sizes)

def gen_non_iid_dataset(data, seed = config.rand_seed, sample_num = config.sample_num, sizes = config.data_distribution):
  engine = Random()
  engine.seed(seed)
  idx = engine.sample(range(len(data)), sample_num)
  idx = [(i, data[i][1]) for i in idx]
  idx.sort(key=lambda ele: ele[1])
  idx = [ele[0] for ele in idx]
  return gen_partitions(data, idx, sizes)

def download(origin, is_gray, train):
  transform = [
    transforms.Resize(224),
    transforms.ToTensor(),
  ]
  if is_gray:
    transform.append(transforms.Lambda(lambda x: x.repeat(3,1,1)))
  transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
  download_args = {
    'root': './fl/data',
    'train': train,
    'download': True,
    'transform': transforms.Compose(transform)
  }
  return origin(**download_args)

def download_datasets(argv):
  if '--cifar' in argv:
    return download(datasets.CIFAR10, is_gray = False, train = True)
  else:
    return download(datasets.MNIST, is_gray = True, train = True)

def download_testsets(argv):
  if '--cifar' in argv:
    return download(datasets.CIFAR10, is_gray = False, train = False)
  else:
    return download(datasets.MNIST, is_gray = True, train = False)

def unlimited_data_loader(dataset, **kwargs):
  datas = iter(DataLoader(dataset, **kwargs)
  while True:
    try:
      yield next(datas)
    except StopIteration:
      datas = iter(DataLoader(dataset, **kwargs))
