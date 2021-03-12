import sys
import torch
import torch.distributed as dist

from functools import reduce
from torch.nn import CrossEntropyLoss
from . import flag, distributed
from .mobilenet import mobilenet_v2
from .communication import recv_model, send_model, send_tensors, recv_tensors
from .utils import clear_params, debug_print, test_accuracy, get_marker
from .data import unlimited_data_loader
from .quantize import QuantizedBuffer, QGD

GLOBAL_EPOCH = 10000
LOCAL_EPOCH = 10
BATCH_SIZE = 4
TARGET_WIDTH = 4

def client_fn(rank, world_size, name, dataset):
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes=10)
  criterion = CrossEntropyLoss()
  param_sizes = map(lambda param: param.data.size(), model.parameters())
  qb_grads = QuantizedBuffer(TARGET_WIDTH, gpu, param_sizes)
  qgd = QGD(model.parameters(), qbuf = qb_grads, lr = lr, momentum=0.4)
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle=True)
  acc_upd = 0

  si = 0.8
  t = 0
  t_m = 10
  buf_error = 0

  model.to(gpu)
  model.train()

  avg_loss = []
  running_loss = 0
  
  for i in range(GLOBAL_EPOCH):
    clear_params(model.parameters(), with_grads=False)
    recv_model(model, with_buffers=False)
    data, label = next(datas)
    output = model(data.to(gpu))
    loss = criterion(output, label.to(gpu))
    running_loss += loss.item()
    loss.backward()
    qgd.step()
    if (i+1)%LOCAL_EPOCH == 0:
      if rank == 1:
        debug_print("Rank %d, 平均损失：%.2lf, scale: %.2lf"%(rank, running_loss/LOCAL_EPOCH, qb_grads.scale))
      avg_loss.append(running_loss/LOCAL_EPOCH)
      running_loss = 0
    m = world_size-1
    grad_diff = qb_grads.diffs_norm
    error = qb_grads.error
    skip_comm = grad_diff <= acc_upd/(lr**2*m**2)+3*(error+buf_error) and t <= t_m
    # skip_comm = ((i+1)%10 != 0)
    if skip_comm:
      dist.send(torch.tensor(False), dst=0)
      t += 1
      buf_error = error
      upd = torch.zeros(1)
      recv_tensors([upd], src=0)
      acc_upd = acc_upd*(t-1)/t+upd/t
    else:
      dist.send(torch.tensor(True), dst=0)
      qb_grads.send_to(0)
      send_tensors(model.buffers(), dst=0)
      qb_grads.reset_buffer()
      recv_tensors(qb_grads.buffers, src=0)
      t = 0
      buf_error = 0
    lr = max(lr*0.9979, min_lr)

def server_fn(rank, world_size, name, testset):
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes=10)

  param_sizes = map(lambda param: param.data.size(), model.parameters())
  qb_grads = [QuantizedBuffer(TARGET_WIDTH, gpu, param_sizes)]
  for _ in range(world_size-2):
    qb_grads.append(QuantizedBuffer(TARGET_WIDTH, gpu, buffers=qb_grads[0].buffers))
  t = [0 for i in range(1, world_size)]

  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  comm_times = []
  uploaded = int(0)
  downloaded = int(0)
  tot_comm = 0

  marker = get_marker()

  model.to(gpu)
  model.eval()
  
  for i in range(GLOBAL_EPOCH):
    # marker('进入循环%d'%i)
    upload_count = 0
    will_recv = torch.tensor(False)
    for j in range(1, world_size):
      downloaded += send_model(model, with_buffers=False, dst=j)
      # marker('向客户端%d传输模型成功'%j)
      dist.recv(will_recv, j)
      if will_recv.item():
        upload_count += 1
        t[j-1] = 0
      else:
        t[j-1] += 1
    tot_comm += upload_count
    # marker('本次循环的客户端通信数量为%d'%upload_count)
    if upload_count != 0:
      alpha = 1/upload_count
      for buffer in model.buffers():
        buffer.fill_(0)
    for j in range(1, world_size):
      if t[j-1] == 0:
        uploaded += qb_grads[j-1].recv_from(j)
        uploaded += recv_tensors(model.buffers(), src=j, alpha=alpha)
    # marker('来自客户端的数据收集完成，正在解码')
    upds = [torch.zeros(param.size()).to(gpu) for param in model.parameters()]
    for j in range(1, world_size):
      if t[j-1] == 0:
        qb_grads[j-1].wait_recv()
        qb_grads[j-1].step()
        # marker('来自客户端%d的数据解码完成'%j)
    # debug_print(qb_grads[0].buffers[-1])
    if upload_count != 0:
      for param, upd, grad in zip(model.parameters(), upds, qb_grads[0].buffers):
        upd.add_(grad, alpha=-lr)
        param.data.add_(grad, alpha=-lr)
    upd = reduce(lambda acc, cur: acc.add_(torch.norm(cur, p=2)), upds, torch.zeros(1).to(gpu))
    for j in range(1, world_size):
      if t[j-1] == 0:
        downloaded += send_tensors(qb_grads[0].buffers, dst=j)
      else:
        downloaded += send_tensors([upd], dst=j)
      # marker('向客户端%d发送数据完成'%j)
    if (i+1)%LOCAL_EPOCH == 0:
      debug_print("训练中...进度：%2.2lf%%"%((i+1)/GLOBAL_EPOCH*100), end=' ')
      downloaded_bytes.append(downloaded)
      uploaded_bytes.append(uploaded)
      accuracy = test_accuracy(model, testset, gpu)
      accuracies.append(accuracy)
      comm_times.append(tot_comm)
      debug_print("总通信次数：%d"%tot_comm, end=' ')
      marker("正确率：%2.2lf%%"%(accuracy*100))
    lr = max(lr*0.9979, min_lr)
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

def train(datasets, testset, is_iid=True):
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