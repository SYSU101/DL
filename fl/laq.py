import torch
import torch.distributed as dist

from ctypes import c_uint8
from .vgg11 import VGG11
from .sgd import FedSGD
from .communication import recv_params, send_params, clear_params
from .data import unlimited_data_loader

GLOBAL_EPOCH = 10000
BATCH_SIZE = 4
BITS_PER_GRAD = 4

def quantize(cur_model, cmp_model, bits_per_grad = BITS_PER_GRAD):
  grad_diffs = []
  for cur_param, cmp_param in zip(cur_model.parameters(), cmp_model.parameters())
    device = cmp_param.grad.device
    cur_grad = cur_param.grad.to(device)
    cmp_grad = cmp_param.grad
    grad_diffs.append(cur_grad.sub(cmp_grad))
  r = 0
  temp = torch.zeros(1)
  for grad_diff in grad_diffs:
    torch.norm(input = grad_diff, p = float('inf'), out = temp)
    r = max(temp.item(), r)
  grid_size = r*2/bits_per_grad
  bits = []
  for grad_diff in grad_diffs:
    grad_diff.apply_(lambda d: ((d+r)/grid_size+0.5)//1)
    grad_diff.apply_(lambda d: (bits.append(int(d)), d*grid_size-r)[-1])
  for cur_param, grad_diff in zip(cur_model.parameters(), grad_diffs):
    device = cur_param.grad.device
    cur_param.grad.copy_(grad_diff.to(device))
  return (bits, r)

def pack_bits(bits, width = BITS_PER_GRAD):
  buffer = bytearray()
  buf_byte = c_uint8(0)
  for seg in bits:
    

def gen_cmp_model():
  cmp_model = VGG11(num_classes = 10)
  clear_params(cmp_model)
  for param in cmp_model.parameters():
    if param.grad != None:
      param.grad.fill_(0)
    else:
      param.grad = torch.zeors(param.data.size)
  return cmp_model

def copy_model(src_model, dst_model):
  param_iter = zip(src_model.parameters(), dst_model.parameters())
  for src_param, dst_param in param_iter:
    dst_device = dst_param.device
    dst_param.data.copy_(src_param.data.to(device))
    dst_param.grad.copy_(src_param.grad.to(device))

def calc_param_diff(cur_params, cmp_params, out, si):
  out.fill_(0)
  temp = torch.zeros(1)
  for cur_param, cmp_param in zip(cur_params, cmp_params):
    torch.norm(cur_param.data.sub(cmp_param.data), p = 2, out = temp)
    out.add_(temp.pow(2))

def client_fn(rank, world_size, dataset):
  device = torch.device('cuda:0')
  model = VGG11(num_classes = 10)
  model.to(device)
  cmp_model = gen_cmp_model()
  criterion = CrossEntropyLoss()
  sgd = FedSGD(model.parameters(), 1e-3, 0.9)
  datas = unlimited_data_loader(dataset, batch_size = BATCH_SIZE, shuffle = True)
  model.train()

  t_m = 10
  si = 0.1
  acc_upd = torch.zeros(1)
  t = 0
  
  for i in range(GLOBAL_EPOCH):
    clear_params(model.parameters())
    recv_params(model.parameters())
    calc_param_diff(model.parameters(), cmp_model.parameters(), acc_upd, si)
    data, label = next(datas)
    output = model(data.to(device))
    loss = criterion(output, label.to(device))
    loss.backward()
    sgd.step()
    bits, r = quantize(model, cmp_model, 4)
    t += 1
