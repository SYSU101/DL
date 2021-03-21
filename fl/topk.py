
import sys
import torch

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
    send_tensors([torch.tensor(len(output), dtype=torch.int64)])
    send_tensors([torch.tensor(output, dtype=torch.float32)])
    send_bytes(pack_bools(valid), dst=0)
    send_tensors(model.buffers())
  save_lists('%s.%d.loss.txt'%(name, rank), avg_loss)

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

  marker = get_marker()

  model.to(gpu)
  model.eval()

  alpha = 1/(world_size-1)

  for i in range(GLOBAL_EPOCH):
    for j in range(1, world_size):
      downloaded += send_model(model, dst=j, with_buffers=False)
    grad_num = torch.zeros(1, dtype=torch.int64)
    for buffer in model.buffers():
      buffer.data.fill_(0)
    for j in range(1, world_size):
      grad_num.fill_(0)
      uploaded += recv_tensors([grad_num], src=j, alpha=1)
      grads = torch.zeros(grad_num)
      uploaded += recv_tensors([grads], src=j)
      bool_bytes = recv_bytes(src=j)
      uploaded += len(bool_bytes)
      valid = unpack_bools(bool_bytes)
      apply_sparse_grads(model.parameters(), grads, valid, lr*alpha)
      uploaded += recv_tensors(model.buffers(), src=j, alpha=alpha)
    lr = max(lr*0.9979, min_lr)
    if (i+1)%LOCAL_EPOCH == 0:
      debug_print("训练中...进度：%2.2lf%%"%((i+1)/GLOBAL_EPOCH*100), end=' ')
      downloaded_bytes.append(downloaded)
      uploaded_bytes.append(uploaded)
      accuracy = test_accuracy(model, testset, gpu)
      accuracies.append(accuracy)
      marker("正确率：%2.2lf%%"%(accuracy*100))
  save_lists('%s.acc.txt'%name,
    accuracies,
    list(range(0, GLOBAL_EPOCH)),
    uploaded_bytes,
    downloaded_bytes
  )

def train(datasets, testset, is_iid=True):
  name = 'Top-k'+('-iid' if is_iid else '-non-iid')
  distributed.simulate(
    server_fn = server_fn,
    server_args = (name, testset),
    client_fn = client_fn,
    gen_client_args = lambda rank: (name, datasets[rank-1])
  )

if __name__ == '__main__':
  launch = flag.parse(sys.argv)
  launch(train)