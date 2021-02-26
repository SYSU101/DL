import torch
import torch.distributed as dist

from ctypes import c_uint32, c_uint64
from struct import pack, unpack
from .vgg11 import VGG11
from .sgd import FedSGD
from .communication import recv_params, send_params, recv_bytes, send_bytes
from .utils import clear_params
from .data import unlimited_data_loader

GLOBAL_EPOCH = 10000
BATCH_SIZE = 4
BITS_PER_GRAD = 4

def quantize(model, ref_grads, bits_per_grad):
  grad_diffs = []
  cpu = torch.device('cpu')
  gpu = torch.device('cuda:0')
  error = 0
  for param, ref_grad in zip(model.parameters(), ref_grads):
    cur_grad = cur_param.grad.to(device)
    grad_diffs.append(cur_grad.sub(ref_grad))
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
    grad_diff.apply_(lambda d: (segs.append(c_uint32(d)), d*grid_size-r)[-1])
    total_diff += grad_diff.norm(p = 2).pow(2).item()
  for cur_param, grad_diff, ref_grads in zip(model.parameters(), grad_diffs, ref_grads):
    ref_grad.add_(grad_diff, alpha = 1)
    error += ref_grad.sub(cur_param.grad).norm(p = 2).pow(2).item()
    cur_param.grad.copy_(ref_grad.to(gpu))
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
  return (segs, grid_size)

def recv_grads(model, width, src = 0):
  grid_size, segs = recv_segs(src)
  segs = iter(segs)
  r = (grid_size*(1<<width))/2
  for param in model.parameters():
    param.grad.apply_(lambda g: g+next(segs)*grid_size-r)

def send_segs(segs, grid_size, width, dst = 0):
  dist.send(torch.tensor([grid_size]), dst)
  buffer = pack_segs(segs, width)
  send_bytes(buffer, dst)

def client_fn(rank, world_size, dataset):
  lr = 8e-5
  gpu = torch.device('cuda:0')
  cpu = torch.device('cpu')
  model = VGG11(num_classes = 10)
  clear_params(model.parameters())
  recv_params(model.parameters())
  model.to(gpu)
  cmp_model = gen_cmp_model()
  criterion = CrossEntropyLoss()
  sgd = FedSGD(model.parameters(), lr, momentum = 0.4)
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle = True)
  model.train()

  si = 0.1
  acc_upd = torch.zeros(1)
  t = 0
  t_mean = torch.zeros(1)
  ref_grads = [param.grad.clone().detach().to(cpu) for param in model.parameters()]
  buf_error = 0
  min_lr = 1e-8
  
  for i in range(GLOBAL_EPOCH):
    data, label = next(datas)
    output = model(data.to(device))
    loss = criterion(output, label.to(device))
    loss.backward()
    sgd.step()
    segs, grid_size, error, grad_diff = quantize(model, ref_grads, BITS_PER_GRAD)
    m = world_size-1
    if grad_diff <= acc_upd.item()/lr**2*m**2+3(error+buf_error) and t < t_mean.item():
      send_segs([], 0, BITS_PER_GRAD)
      t += 1
    else:
      send_segs(segs, grid_size, BITS_PER_GRAD)
      acc_upd.data.fill_(0)
    buf_error = error
    segs, grid_size = recv_segs()
    dist.recv(t_mean)
    r = BITS_PER_GRAD*segs/2
    segs = iter(segs)
    for param, ref_grad in zip(model.parameters(), ref_grads):
      ref_grad.apply_(lambda g: g+next(segs)*grid_size-r)
      param.grad.copy_(ref_grad.to(gpu))
      param.data.add_(param.grad, alpha = lr)
      acc_upd.add(ref_grad.norm(p = 2).pow(2), alpha = si)
    lr = max(lr*0.997, min_lr)
