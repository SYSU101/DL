#! /usr/bin/env python3.8
from torchvision import datasets, transforms
from .data import download_datasets, download_testsets, gen_iid_dataset, gen_non_iid_dataset

def parse(argv):
  dataset = download_datasets(argv)
  testset = download_testsets(argv)
  is_non_iid = '--non-iid' in argv
  if is_non_iid:
    datasets = gen_non_iid_dataset(dataset)
  else:
    datasets = gen_iid_dataset(dataset)
  def launch(train):
    train(datasets, testset, not is_non_iid)
  return launch
