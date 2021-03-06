import gc
import torch
import torch.cuda
import torch.distributed as dist
from .utils import debug_print
from torch import Tensor, flatten, torch
from struct import pack, unpack
from string import Template
from ctypes import c_uint64

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

def send_segs(segs, scale, width, dst = 0):
  dist.send(torch.tensor([scale]), dst)
  buffer = pack_segs(segs, width)
  send_bytes(buffer, dst)
  return 4+len(buffer)

def recv_segs(src, width):
  scale = torch.zeros(1)
  dist.recv(scale, src)
  scale = scale.item()
  buffer = recv_bytes(src)
  segs = unpack_segs(buffer, width)
  return (segs, scale, 4+len(buffer))