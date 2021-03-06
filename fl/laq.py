import sys
import torch
import torch.distributed as dist

from functools import reduce
from torch.nn import CrossEntropyLoss
from threading import Thread
from . import flag, distributed
from .mobilenet import mobilenet_v2
from .sgd import FedSGD
from .communication import recv_params, send_params
from .utils import clear_params, debug_print, test_accuracy
from .data import unlimited_data_loader
from .quantize import QuantizedBuffer
from time import time

GLOBAL_EPOCH = 10000
LOCAL_EPOCH = 100
BATCH_SIZE = 4
TARGET_WIDTH = 4

def calc_update(params, out, lr, si):
  out.fill_(0)

def client_fn(rank, world_size, name, dataset):
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes = 10)
  criterion = CrossEntropyLoss()
  sgd = FedSGD(model.parameters(), lr, momentum = 0.4)
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle = True)

  model.to(gpu)
  clear_params(model.parameters())
  recv_params(model.parameters())
  model.train()

  si = 0.1
  acc_upd = torch.zeros(1)
  t = 0
  t_mean = torch.zeros(1)
  buf_error = 0
  buffer_num = reduce(lambda count, _: count+1, model.buffers(), 0)
  param_num = reduce(lambda count, _: count+1, model.parameters(), 0)
  qb_buffers = QuantizedBuffer(buffer_num, TARGET_WIDTH, gpu)
  qb_grads = QuantizedBuffer(param_num, TARGET_WIDTH, gpu)

  avg_loss = []
  running_loss = 0
  
  for i in range(GLOBAL_EPOCH):
    data, label = next(datas)
    output = model(data.to(gpu))
    loss = criterion(output, label.to(gpu))
    running_loss += loss.item()
    loss.backward()
    if (i+1)%LOCAL_EPOCH == 0:
      avg_loss.append(running_loss/LOCAL_EPOCH)
      running_loss = 0
    sgd.step()
    qb_buffers.update(model.buffers())
    qb_grads.update(map(lambda param: param.grad, model.parameters()))
    m = world_size-1
    grad_diff = qb_grads.diffs_norm
    error = qb_grads.error+qb_buffers.error
    if grad_diff <= acc_upd.item()/(lr**2*m**2)+3*(error+buf_error) and t < t_mean.item():
      dist.send(torch.tensor(False), dst = 0)
      t += 1
    else:
      dist.send(torch.tensor(True), dst = 0)
      qb_grads.send_to(0)
      qb_buffers.send_to(0)
      t = 0
      acc_upd.data.fill_(0)
    buf_error = error
    qb_grads.recv_from(0)
    qb_buffers.recv_from(0)
    dist.recv(t_mean)
    qb_grads.step(map(lambda param: param.grad, model.parameters()))
    qb_buffers.step(model.buffers())
    temp = torch.zeros(1)
    for param in params:
      param.grad.mul_(lr)
      torch.norm(param.grad, p = 2, out = temp)
      acc_upd.add_(temp.pow_(2), alpha = si)
      param.data.add_(param.grad)
    acc_upd.add(qb_buffers.diffs_norm, alpha = si)
    lr = max(lr*0.9979, min_lr)

def server_fn(rank, world_size, name, testset):
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes = 10)

  buffer_num = reduce(lambda count, _: count+1, model.buffers(), 0)
  param_num = reduce(lambda count, _: count+1, model.parameters(), 0)
  qb_buffers = QuantizedBuffer(buffer_num, TARGET_WIDTH, gpu)
  qb_grads = QuantizedBuffer(param_num, TARGET_WIDTH, gpu)
  t = [0 for i in range(1, world_size)]

  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  uploaded = int(0)
  downloaded = int(0)

  for param in model.parameters():
    param.grad = param.data.clone().detach().fill_(0)
  model.to(gpu)
  for i in range(1, world_size):
    downloaded += send_params(model.parameters(), dst = i)
  
  for i in range(GLOBAL_EPOCH):
    debug_print("循环%d："%i)
    last_check_point = time()
    if (i+1)%LOCAL_EPOCH == 0:
      debug_print("训练中...进度：%2.2lf%%"%(i/GLOBAL_EPOCH*100), end = ' ')
    upload_count = 0
    will_recv = torch.tensor(False)
    for j in range(1, world_size):
      dist.recv(will_recv, j)
      if will_recv.item():
        upload_count += 1
        t[j-1] = 0
      else:
        t[j-1] += 1
    now = time()
    debug_print("上传客户端数量确认完成，用时%.2lf秒"%(now-last_check_point))
    last_check_point = now
    t_mean = reduce(lambda acc, cur: acc+cur/(world_size-1), t, 0)
    t_mean = torch.tensor(t_mean)
    alpha = 1/upload_count
    for j in range(1, world_size):
      if t[j-1] == 0:
        uploaded += qb_grads.recv_from(j)
        uploaded += qb_buffers.recv_from(j)
        now = time()
        debug_print("来自客户机%d的数据接收完成，用时%.2lf秒"%(j, now-last_check_point))
        last_check_point = now
        qb_grads.step(map(lambda param: param.grad, model.parameters()), alpha)
        qb_buffers.step(model.buffers(), alpha)
        now = time()
        debug_print("来自客户机%d的数据计算完成，用时%.2lf秒"%(j, now-last_check_point))
        last_check_point = now
    qb_grads.update(map(lambda param: param.grad, model.parameters()))
    qb_buffers.update(model.buffers())
    now = time()
    debug_print("重新量化完成，用时%.2lf秒"%(j, now-last_check_point))
    last_check_point = now
    for j in range(1, world_size):
      downloaded += qb_grads.send_to(j)
      downloaded += qb_buffers.send_to(j)
      dist.send(t_mean, dst = j)
      downloaded += 8
      now = time()
      debug_print("数据发送到客户机%d完成，用时%.2lf秒"%(j, now-last_check_point))
      last_check_point = now
    for param in model.parameters():
      param.add_(param.grad, alpha = lr)
    now = time()
    debug_print("梯度应用完成，用时%.2lf秒"%(now-last_check_point))
    last_check_point = now
    if (i+1)%LOCAL_EPOCH == 0:
      downloaded_bytes.append(downloaded)
      uploaded_bytes.append(uploaded)
      accuracy = test_accuracy(model, testset, gpu)
      accuracies.append(accuracy)
      now = time()
      debug_print("正确率计算完成，用时%.2lf秒"%(now-last_check_point))
      last_check_point = now
      debug_print("正确率：%2.2lf%%"%(accuracy*100))
    lr = max(lr*0.997, min_lr)
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

def train(datasets, testset, is_iid = True):
  name = 'LAQ'+('-iid' if is_iid else '-non-iid')
  distributed.simulate(
    server_fn = server_fn,
    server_args = (name, testset),
    client_fn = client_fn,
    gen_client_args = lambda rank: (name, datasets[rank-1])
  )

if __name__ == '__main__':
  launch = flag.parse(sys.argv)
  launch(train)