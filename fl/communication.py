import gc
import torch
import torch.cuda
import torch.distributed as dist
from torch import Tensor, flatten, torch
from struct import pack, unpack
from string import Template
from ctypes import c_uint64
from multiprocessing import Process, Pipe

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

EOP = '__END_OF_PIPE__'

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

def send_tensors(tensors, dst = 0):
  buffer = bytearray()
  for tensor in tensors:
    tensor2bytes(param.data, buffer)
  send_bytes(buffer, dst)
  return len(buffer)

def recv_tensors(tensors, src = 0, alpha = 1.0):
  buffer = recv_bytes(src)
  size = len(buffer)
  for tensor in tensors:
    device = tensor.data.device
    buffer, recv = bytes2tensor(buffer, tensor.data.size(), tensor.data.dtype)
    tensor.data.add_(recv.to(device), alpha = alpha)

def send_model(model, dst = 0, with_grads = False):
  size = 0
  size += send_tensors(model.parameters(), dst)
  if with_grads:
    size += send_tensors(map(lambda p: p.grad, model.parameters()), dst)
  size += send_tensors(model.buffers(), dst)
  return size

def recv_model(model, src = 0, with_grads = False, alpha = 1.0):
  size = 0
  size += recv_tensors(src, alpha)
  if with_grads:
    size += recv_tensors(map(lambda p: p.grad, model.parameters()), src, alpha)
  size += recv_tensors(model.buffers(), src, alpha)
  return size
  
def pack_segs(segs, width):
  buffer = bytearray()
  rest_len = 0
  rest_byte = 0
  byte_buffer = 0
  for seg in segs:
    pack_len = rest_len+width
    seg = ((seg << (64-pack_len)) | rest_byte)
    byte_num = pack_len//8
    rest_len = pack_len%8
    buffer.extend(pack('>Q', c_uint64(seg).value)[:byte_num])
    rest_byte = c_uint64(seg << (byte_num*8)).value
  if rest_len > 0:
    buffer.extend(pack('>Q', c_uint64(rest_byte).value)[:1])
  return buffer

def unpack_segs(segs, buffer, width):
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
    segs.send(seg)
  segs.send(EOP)
  segs.close()

def send_segs(segs, scale, width, dst = 0):
  buffer = pack_segs(segs, width)
  dist.send(torch.tensor([scale]), dst)
  send_bytes(buffer, dst)
  return 4+len(buffer)

def recv_segs(src, width):
  scale = torch.zeros(1)
  dist.recv(scale, src)
  scale = scale.item()
  buffer = recv_bytes(src)
  recv_conn, send_conn = Pipe(duplex = False)
  p = Process(target = unpack_segs, args = (send_conn, buffer, width))
  p.start()
  return (recv_conn, scale, 4+len(buffer), p)