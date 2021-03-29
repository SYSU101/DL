
import sys
import torch
import torch.distributed as dist

from functools import reduce
from torch.nn import CrossEntropyLoss
from . import flag, distributed
from .communication import *
from .mobilenet import mobilenet_v2
from .utils import *
from .data import unlimited_data_loader
from .sparse import Sparser, SparseSGD, apply_sparse_grads

GLOBAL_EPOCH = 10000
LOCAL_EPOCH = 10
BATCH_SIZE = 4
BITS_PER_GRAD = 4

def client_fn(rank, world_size, name, dataset):
  cpu = torch.device('cpu')
  gpu = torch.device('cuda:0')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes=10)
  criterion = CrossEntropyLoss()
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle=True)
  param_num = reduce(lambda acc, cur: acc+cur.numel(), model.parameters(), 0)
  sparser = Sparser(p=0.01, tot_num=param_num)
  sgd = SparseSGD(model.parameters(), sparser, lr, momentum=0.4)

  model.to(gpu)
  model.train()

  avg_loss = []
  running_loss = 0
  upds = []
  iupds = 0
  t_m = 10
  alpha = 1
  beta = sqrt(1/(2-alpha))
  si = 0.8

  for i in range(GLOBAL_EPOCH):
    clear_params(model.parameters(), with_grads=False)
    recv_model(model, with_buffers=False)
    data, label = next(datas)
    output = model(data.to(gpu))
    loss = criterion(output, label.to(gpu))
    running_loss += loss.item()
    loss.backward()
    if (i+1)%LOCAL_EPOCH == 0:
      if rank == 1:
        debug_print("Rank %d, 平均损失：%.2lf"%(rank, running_loss/LOCAL_EPOCH))
      avg_loss.append(running_loss/LOCAL_EPOCH)
      running_loss = 0
    sparser.reset()
    sgd.step()
    output, valid = sparser.output()
    mags = [abs(out) for out in output]
    threshold = min(mags)
    radius = max(mags)-threshold
    grid_size = radius*2/((1<<BITS_PER_GRAD)-1)
    qnorm = 0
    for i in range(len(output)):
      if output[i] < 0:
        output[i] = int((output[i]+threshold+radius)/grid_size+0.5)
        qnorm += (output[i]*grid_size-radius-threshold)**2
      else:
        output[i] = int((output[i]-threshold+radius)/grid_size+0.5)
        qnorm += (output[i]*grid_size-radisu+threshold)**2
    upload_cond = 0 if len(upds) == 0 else sum(upds)*si/len(upds)
    upload_cond *= beta*(beta-alpha)/(2*((1-beta)**2)*(lr**2))
    if qnorm < upload_cond:
      dist.send(torch.tensor(True), 0)
    else:
      dist.send(torch.tensor(False), 0)
      threshold = torch.tensor(threshold, dtype=torch.float64)
      grid_size = torch.tensor(grid_size, dtype=torch.float64)
      send_tensors([threshold, grid_size])
      send_bytes(pack_segs(output, BITS_PER_GRAD), dst=0)
      send_bytes(pack_bools(valid), dst=0)
      send_tensors(model.buffers())
    upd = torch.zeros(0, dtype=torch.float64)
    recv_tensors([upd])
    if len(upds) >= t_m:
      upds[iupds] = upd.item()
      iupds = (iupds+1)%t_m
    else:
      upds.append(upd.item())

def server_fn(rank, world_size, name, testset):
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')

  lr = 1e-3
  min_lr = 1e-5
  model = mobilenet_v2(num_classes=10)

  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  uploaded = int(0)
  downloaded = int(0)
  tot_comm = 0
  comm_times = []

  marker = get_marker()

  model.to(gpu)
  model.eval()

  alpha = 1/(world_size-1)
  threshold = torch.zeros(1, dtype=torch.float64)
  grid_size = torch.zeros(1, dtype=torch.float64)
  skip_comm = [torch.tensor(False) for _ in range(1, world_size)]
  comm_num = 0

  for i in range(GLOBAL_EPOCH):
    for j in range(1, world_size):
      downloaded += send_model(model, dst=j, with_buffers=False)
    comm_num = 0
    for j in range(1, world_size):
      dist.recv(skip_comm[j-1], j)
      if not skip_comm[j-1].item():
        comm_num += 1
    alpha = 1/comm_num
    tot_comm += comm_num
    upd = 0
    for buffer in model.buffers():
      buffer.data.fill_(0)
    for j in range(1, world_size):
      if not skip_comm[j-1].item():
        threshold.fill_(0)
        grid_size.fill_(0)
        uploaded += recv_tensors([threshold, grid_size], src=j, alpha=1)
        segs = recv_bytes(src=j)
        uploaded += len(segs)
        segs = unpack_segs_sync(segs, BITS_PER_GRAD)
        bool_bytes = recv_bytes(src=j)
        uploaded += len(bool_bytes)
        valid = unpack_bools(bool_bytes)
        radius = grid_size*((1<<BITS_PER_GRAD)-1)
        for i in range(len(segs)):
          if segs[i] < (1<<(BITS_PER_GRAD-1)):
            segs[i] = segs[i]*grid_size.item()-radius-threshold.item()
          else:
            segs[i] = segs[i]*grid_size.item()-radius+threshold.item()
          upd += (segs[i]*lr*alpha)**2
        apply_sparse_grads(model.parameters(), segs, valid, lr*alpha)
        uploaded += recv_tensors(model.buffers(), src=j, alpha=alpha)
    upd = torch.tensor(upd, dtype=torch.float64)
    for j in range(1, world_size):
      download += send_tensors([upd], dst=j)
    lr = max(lr*0.9979, min_lr)
    if (i+1)%LOCAL_EPOCH == 0:
      debug_print("训练中...进度：%2.2lf%%"%((i+1)/GLOBAL_EPOCH*100), end=' ')
      downloaded_bytes.append(downloaded)
      uploaded_bytes.append(uploaded)
      accuracy = test_accuracy(model, testset, gpu)
      accuracies.append(accuracy)
      comm_times.append(tot_comm)
      marker("正确率：%2.2lf%%"%(accuracy*100))
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

def train(datasets, testset, is_iid=True):
  name = 'TLQ'+('-iid' if is_iid else '-non-iid')
  distributed.simulate(
    server_fn = server_fn,
    server_args = (name, testset),
    client_fn = client_fn,
    gen_client_args = lambda rank: (name, datasets[rank-1])
  )

if __name__ == '__main__':
  launch = flag.parse(sys.argv)
  launch(train)