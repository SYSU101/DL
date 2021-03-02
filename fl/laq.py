import sys
import torch
import torch.distributed as dist

from ctypes import c_uint32, c_uint64
from struct import pack, unpack
from functools import reduce
from torch.nn import CrossEntropyLoss
from . import flag, distributed
from .mobilenet import mobilenet_v2
from .sgd import FedSGD
from .communication import recv_params, send_params, recv_bytes, send_bytes
from .utils import clear_params, debug_print, test_accuracy, get_params, get_grads
from .data import unlimited_data_loader

GLOBAL_EPOCH = 10000
LOCAL_EPOCH = 100
BATCH_SIZE = 4
BITS_PER_GRAD = 4

def quantize(model, ref_grads, bits_per_grad):
  grad_diffs = []
  cpu = torch.device('cpu')
  gpu = torch.device('cuda:0')
  error = 0
  for grad, ref_grad in zip(get_grads(model), ref_grads):
    grad = grad.to(cpu)
    grad_diffs.append(grad.sub(ref_grad))
  r = 0
  temp = torch.zeros(1)
  for grad_diff in grad_diffs:
    torch.norm(input = grad_diff, p = float('inf'), out = temp)
    r = max(temp.item(), r)
  grid_size = r*2/bits_per_grad
  segs = []
  total_diff = 0
  for grad_diff in grad_diffs:
    grad_diff.apply_(lambda d: ((d+r)/grid_size+0.5)//1)
    grad_diff.apply_(lambda d: (segs.append(c_uint32(int(d))), d*grid_size-r)[-1])
    total_diff += grad_diff.norm(p = 2).pow_(2).item()
  for cur_grad, grad_diff, ref_grads in zip(get_grads(model), grad_diffs, ref_grads):
    ref_grad.add_(grad_diff, alpha = 1)
    error += ref_grad.sub(cur_param.grad).norm(p = 2).pow_(2).item()
    cur_grad.copy_(ref_grad.to(gpu))
  return (segs, grid_size, error, total_diff)

def pack_segs(segs, width):
  buffer = bytearray()
  rest_len = 0
  rest_byte = 0
  byte_buffer = 0
  for seg in segs:
    seg = seg.value
    pack_len = rest_len+width
    seg = ((seg << (64-pack_len)) | rest_byte)
    byte_num = pack_len//8
    rest_len = pack_len%8
    buffer.extend(pack('>Q', c_uint64(seg).value)[:byte_num])
    rest_byte = (seg << (byte_num*8))
  if rest_len > 0:
    buffer.extend(pack('>Q', c_uint64(rest_byte).value)[:1])
  return buffer

def unpack_segs(buffer, width):
  segs = []
  last_rest = 0
  while len(buffer)*8 >= width+last_rest:
    cur_rest = width
    seg = 0
    while cur_rest > 0:
      msb = 8-last_rest
      lsb = max(8-last_rest-cur_rest, 0)
      byte = unpack('>B', buffer[:1])[0]
      byte &= ((1 << msb)-1)
      byte >>= lsb
      seg = (seg << min(8, cur_rest)) | byte
      cur_rest -= (8-last_rest)
      if lsb == 0:
        last_rest = 0
        buffer = buffer[1:]
      else:
        last_rest = 8-lsb
    segs.append(seg)
  return segs

def calc_update(params, out, si):
  out.fill_(0)
  temp = torch.zeros(1)
  for param in params:
    torch.norm(param.grad, p = 2, out = temp)
    out.add_(temp.pow(2), alpha = si)

def recv_segs(src, width):
  grid_size = torch.zeros(1)
  dist.recv(grid_size, src)
  grid_size = grid_size.item()
  buffer = recv_bytes()
  segs = unpack_segs(buffer, width)
  return (segs, grid_size, 4+len(buffer))

def recv_grads(model, width, src = 0):
  grid_size, segs = recv_segs(src)
  segs = iter(segs)
  r = (grid_size*(1<<width))/2
  for grad in get_grads(model):
    grad.apply_(lambda g: g+next(segs)*grid_size-r)

def send_segs(segs, grid_size, width, dst = 0):
  dist.send(torch.tensor([grid_size]), dst)
  buffer = pack_segs(segs, width)
  send_bytes(buffer, dst)
  return 4+len(buffer)

def client_fn(rank, world_size, name, dataset):
  lr = 1e-3
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')
  model = mobilenet_v2(num_classes = 10)
  clear_params(model.parameters())
  recv_params(model.parameters())
  model.to(gpu)
  avg_loss = []
  criterion = CrossEntropyLoss()
  sgd = FedSGD(model.parameters(), lr, momentum = 0.4)
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle = True)
  model.train()

  si = 0.1
  acc_upd = torch.zeros(1)
  t = 0
  t_mean = torch.zeros(1)
  ref_grads = [param.clone().detach().to(cpu).fill_(0) for param in get_params(model)]
  buf_error = 0
  min_lr = 1e-5
  
  for i in range(GLOBAL_EPOCH):
    data, label = next(datas)
    output = model(data.to(gpu))
    loss = criterion(output, label.to(gpu))
    loss.backward()
    if (i+1)%LOCAL_EPOCH == 0:
      avg_loss.append(loss.item())
    sgd.step()
    segs, grid_size, error, grad_diff = quantize(model, ref_grads, BITS_PER_GRAD)
    m = world_size-1
    if grad_diff <= acc_upd.item()/lr**2*m**2+3(error+buf_error) and t < t_mean.item():
      send_segs([], 0, BITS_PER_GRAD)
      t += 1
    else:
      send_segs(segs, grid_size, BITS_PER_GRAD)
      t = 0
      acc_upd.data.fill_(0)
    buf_error = error
    segs, grid_size = recv_segs()
    dist.recv(t_mean)
    r = BITS_PER_GRAD*segs/2
    segs = iter(segs)
    iter_ref_grads = iter(ref_grads)
    for param, ref_grad in zip(model.parameters(), iter_ref_grads):
      ref_grad.apply_(lambda g: g+next(segs)*grid_size-r)
      param.grad.copy_(ref_grad.to(gpu))
      param.data.add_(param.grad, alpha = lr)
      acc_upd.add(ref_grad.norm(p = 2).div_(lr).pow_(2), alpha = si)
    for buffer, ref_grad in zip(model.buffers(), iter_ref_grads):
      ref_grad.apply_(lambda g: g+next(segs)*grid_size-r)
      buffer.copy_(ref_grad.to(gpu))
      acc_upd.add(ref_grad.norm(p = 2).div_(lr).pow_(2), alpha = si)
    lr = max(lr*0.9979, min_lr)

def server_fn(rank, world_size, name, testset):
  uploaded_bytes = []
  downloaded_bytes = []
  accuracies = []
  uploaded = int(0)
  downloaded = int(0)

  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')
  model = mobilenet_v2(num_classes = 10)
  model.to(gpu)
  lr = 0.1
  min_lr = 1e-8

  for i in range(1, world_size):
    downloaded += send_params(model.parameters(), dst = i)
  t = [0 for i in range(1, world_size)]
  ref_grads = [param.clone().detach().to(cpu).fill_(0) for param in get_params(model)]
  segs = [None for _ in range(1, world_size)]
  grid_sizes = [0 for _ in range(1, world_size)]
  
  for i in range(GLOBAL_EPOCH):
    debug_print("训练中...进度：%2.4lf%%"%(i/GLOBAL_EPOCH*100), end = ' ')
    upload_count = 0
    for j in range(world_size-1):
      segs[j], grid_sizes[j], bytes_count = recv_segs(src = j+1, width = BITS_PER_GRAD)
      uploaded += bytes_count
      if len(segs[j]) > 0:
        upload_count += 1
        t[j] = 0
      else:
        t[j] += 1
    t_mean = reduce(lambda acc, cur: acc+cur/(world_size-1), t)
    for j in range(world_size-1):
      r = BITS_PER_GRAD*grid_sizes[j]
      iseg = iter(segs[j])
      for grad in get_grads(model):
        grad.to(cpu).apply_(lambda d: (next(iseg)*grid_sizes[j]-r)/upload_count+d)
    agg_segs, grid_size, _, _ = quantize(model, ref_grads, BITS_PER_GRAD)
    t_mean = torch.tensor(t_mean)
    for j in range(1, world_size):
      downloaded += send_segs(agg_segs, grid_size, BITS_PER_GRAD, dst=j)
      dist.send(t_mean, dst = j)
    for param, ref_grad in zip(model.parameters(), ref_grads):
      ref_grad.copy_(param.grad)
      param.data.add_(param.gard, alpha=lr)
    if (i+1)%LOCAL_EPOCH == 0:
      downloaded_bytes.append(downloaded)
      uploaded_bytes.append(uploaded)
      accuracy = test_accuracy(model, testset, gpu)
      accuracies.append(accuracy)
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