#! /usr/bin/env python3.8
import sys
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from . import flag, config, distributed
from .communication import send_params, recv_params
from .utils import test_accuracy, debug_print, save_lists
from .vgg11 import VGG11

LOCAL_EPOCH = 10
GLOBAL_EPOCH = 1000
BATCH_SIZE = 4

def clear_params(params):
  for param in params:
    param.data.fill_(0)

def client_fn(rank, world_size, dataset):
  device = torch.device('cuda:0')
  model = VGG11(num_classes = 10)
  model.to(device)
  criterion = CrossEntropyLoss()
  sgd = SGD(model.parameters(), 1e-3, 0.9)
  model.train()
  for i in range(GLOBAL_EPOCH):
    clear_params(model.parameters())
    recv_params(model.parameters())
    datas = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    for j in range(LOCAL_EPOCH):
      for data, target in datas:
        output = model(data.to(device))
        loss = criterion(output.to(device), target.to(device))
        loss.backward()
        sgd.step()
    send_params(model.parameters())

def server_fn(rank, world_size, testset):
  global uploaded_bytes
  global downloaded_bytes
  global accuracies
  device = torch.device('cuda:0')
  model = VGG11(num_classes = 10)
  model.to(device)
  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  uploaded = int(0)
  downloaded = int(0)
  for i in range(GLOBAL_EPOCH):
    debug_print("训练中...进度：%2.2lf%%"%(i/GLOBAL_EPOCH*100), end = ' ')
    for j in range(1, world_size):
      downloaded += send_params(params = model.parameters(), dst = j)
    clear_params(model.parameters())
    for j in range(1, world_size):
      uploaded += recv_params(
        params = model.parameters(),
        alpha = config.data_distribution[j-1],
        src = j
      )
    uploaded_bytes.append(uploaded)
    downloaded_bytes.append(downloaded)
    accuracy = test_accuracy(model, testset, device)
    accuracies.append(accuracy)
    debug_print("正确率：%2.2lf%%"%(accuracy*100))

def train(datasets, testset, is_iid = True):
  name = 'FedAvg'.join('-iid' if is_iid else '-non-iid')
  distributed.simulate(
    server_fn = server_fn,
    server_args = (testset,),
    client_fn = client_fn,
    gen_client_args = lambda rank: (datasets[rank],)
  )
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

if __name__ == '__main__':
  launch = flag.parse(sys.argv)
  launch(train)
