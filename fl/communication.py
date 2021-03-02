import gc
import torch
import torch.cuda
import torch.distributed as dist
from .utils import debug_print
from torch import Tensor, flatten, torch
from struct import pack, unpack
from string import Template

fmt_code = {
  torch.float32: Template('>${num}f'),
  torch.float64: Template('>${num}d'),
  torch.int64: Template('>${num}q'),
}

byte_size = {
  torch.float32: 4,
  torch.float64: 8,
  torch.int64: 8,
}

def get_fmt_code(dtype):
  if not dtype in fmt_code:
    raise RuntimeError('dtype %s is not supported'%dtype)
  return fmt_code.get(dtype)

def get_byte_size(dtype):
  if not dtype in byte_size:
    raise RuntimeError('dtype %s is not supported'%dtype)
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
  buffer = torch.zeros(size.item(), dtype=torch.uint8)
  dist.recv(buffer, src)
  return bytes(buffer.tolist())

def isend_bytes(buffer, dst):
  buffer = torch.tensor(buffer, dtype=torch.uint8)
  size = torch.tensor([buffer.size().numel()], dtype=torch.int64)
  dist.send(size, dst)
  return dist.isend(buffer, dst)

def send_params_(params, dst, with_grad, model_buffer):
  buffer = bytearray()
  for param in params:
    tensor2bytes(param.data, buffer)
    if with_grad:
      tensor2bytes(param.grad, buffer)
  if model_buffer != None:
    for buf in model_buffer:
      tensor2bytes(buf.data, buffer)
  send_bytes(buffer, dst)
  return len(buffer)

def send_params(params, dst = 0, with_grad = False, model_buffer = None):
  size = send_params_(params, dst, with_grad, model_buffer)
  gc.collect()
  return size

def recv_params_(params, alpha, src, with_grad, model_buffer):
  buffer = recv_bytes(src)
  size = len(buffer)
  for param in params:
    device = param.data.device
    buffer, local_param = bytes2tensor(buffer, param.data.size(), param.data.dtype)
    param.data.add_(local_param.to(device), alpha = alpha)
    if with_grad:
      buffer, local_grad = bytes2tensor(buffer, param.grad.size(), param.grad.dtpye)
      param.grad.copy_(local_grad)
  if model_buffer != None:
    for buf in model_buffer:
      device = buf.data.device
      buffer, recv_buf = bytes2tensor(buffer, buf.data.size(), buf.data.dtype)
      buf.add_(recv_buf.to(device), alpha = alpha)
  return size

def recv_params(params, alpha = 1.0, src = 0, with_grad = False, model_buffer = None):
  size = recv_params_(params, alpha, src, with_grad, model_buffer)
  torch.cuda.empty_cache()
  gc.collect()
  return size