import gc
import torch
import torch.cuda
import torch.distributed as dist
from torch import Tensor, flatten, torch
from struct import pack, unpack
from string import Template

fmt_code = {
  torch.float32: Template('>${num}f'),
  torch.float64: Template('>${num}d'),
}

byte_size = {
  torch.float32: 4,
  torch.float64: 8,
}

def get_fmt_code(dtype):
  if not dtype in fmt_code:
    raise RuntimeError('dtype %s is not supported'%tensor.dtype)
  return fmt_code.get(dtype)

def get_byte_size(dtype):
  if not dtype in byte_size:
    raise RuntimeError('dtype %s is not supported'%tensor.dtype)
  return byte_size.get(dtype)

def tensor2bytes(tensor, buffer):
  fmt = get_fmt_code(tensor.dtype).substitute(num = '')
  for num in tensor.flatten().tolist():
    buffer.extend(pack(fmt, num))

def bytes2tensor(buffer, size, dtype):
  bsize = get_byte_size(dtype)
  num = size.numel()
  fmt = get_fmt_code(dtype).substitute(num = num)
  buffer, rest = buffer[:num*bsize], buffer[num*bsize:]
  tensor = Tensor(unpack(fmt, buffer))
  tensor.resize_(size)
  return (rest, tensor)

def send_bytes(buffer, dst):
  buffer = torch.tensor(buffer, dtype=torch.uint8)
  size = torch.tensor([buffer.size().numel()], dtype=torch.int64)
  dist.send(size, dst)
  dist.send(buffer, dst)

def recv_bytes(src):
  size = torch.zeros(1, dtype=torch.int64)
  dist.recv(size, src)
  buffer = torch.zeros(size, dtype=torch.uint8)
  dist.recv(buffer, src)
  return bytes(buffer.tolist())

def isend_bytes(buffer, dst):
  buffer = torch.tensor(buffer, dtype=torch.uint8)
  size = torch.tensor([buffer.size().numel()], dtype=torch.int64)
  dist.send(size, dst)
  return dist.isend(buffer, dst)

def send_params_(params, dst):
  buffer = bytearray()
  for param in params:
    tensor2bytes(param.data, buffer)
  send_bytes(buffer, dst)
  return len(buffer)

def send_params(params, dst = 0):
  size = send_params_(params, dst)
  gc.collect()
  return size

def recv_params_(params, alpha, src, device):
  buffer = recv_bytes(src)
  for param in params:
    buffer, local_param = bytes2tensor(buffer, param.size(), param.dtype)
    param.data.add_(local_param.to(device), alpha = alpha)
  return len(buffer)

def recv_params(params, alpha = 1.0, src = 0, device = torch.device('cuda:0')):
  size = recv_params_(params, alpha, src, device)
  torch.cuda.empty_cache()
  gc.collect()
  return size