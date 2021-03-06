#! /usr/bin/env python3.8
import sys
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from . import flag, config, distributed
from .communication import send_model, recv_model
from .utils import test_accuracy, debug_print, save_lists, decay_learning_rate, clear_params
from .mobilenet import mobilenet_v2

LOCAL_EPOCH = 10
GLOBAL_EPOCH = 1000
BATCH_SIZE = 4

def client_fn(rank, world_size, name, dataset):
  device = torch.device('cuda:0')
  model = mobilenet_v2(num_classes=10)
  model.to(device)
  criterion = CrossEntropyLoss()
  sgd = SGD(model.parameters(), lr = 1e-3, momentum=0.4)
  model.train()
  avg_loss = []
  for i in range(GLOBAL_EPOCH):
    clear_params(model.parameters(), model.buffers())
    recv_model(model)
    datas = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
    running_loss = 0
    for j in range(LOCAL_EPOCH):
      for data, target in datas:
        output = model(data.to(device))
        loss = criterion(output.to(device), target.to(device))
        running_loss += loss.item()
        loss.backward()
        sgd.step()
      decay_learning_rate(sgd, alpha = 0.9, min_lr=1e-5)
    avg_loss.append(running_loss/(j*len(datas)))
    send_model(model)
  save_lists('%s.%d.loss.txt'%(name, rank), avg_loss)

def server_fn(rank, world_size, name, testset):
  device = torch.device('cuda:0')
  model = mobilenet_v2(num_classes=10)
  model.to(device)
  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  uploaded = int(0)
  downloaded = int(0)
  for i in range(GLOBAL_EPOCH):
    debug_print("训练中...进度：%2.2lf%%"%(i/GLOBAL_EPOCH*100), end=' ')
    for j in range(1, world_size):
      downloaded += send_model(model, dst=j)
    clear_params(model.parameters(), model.buffers())
    for j in range(1, world_size):
      uploaded += recv_model(
        model, 
        alpha = config.data_distribution[j-1],
        src = j,
      )
    uploaded_bytes.append(uploaded)
    downloaded_bytes.append(downloaded)
    accuracy = test_accuracy(model, testset, device)
    accuracies.append(accuracy)
    debug_print("正确率：%2.2lf%%"%(accuracy*100))
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

def train(datasets, testset, is_iid=True):
  name = 'FedAvg'+('-iid' if is_iid else '-non-iid')
  distributed.simulate(
    server_fn = server_fn,
    server_args = (name, testset),
    client_fn = client_fn,
    gen_client_args = lambda rank: (name, datasets[rank-1])
  )

if __name__ == '__main__':
  launch = flag.parse(sys.argv)
  launch(train)
