import torch
import torch.distributed as dist
from sys import getsizeof
from ctypes import c_uint32
from threading import Thread
from torch.optim.optimizer import Optimizer, required
from .communication import send_bytes, pack_segs, recv_segs, EOP
from .utils import debug_print

class QuantizedBuffer(object):
  def __init__(self, sizes, data_width, device):
    self.buffers = [torch.zeros(size).to(device) for size in sizes]
    self.device = device
    self.data_width = data_width
    self.scale = 0
    self.result = []
    self.error = 0
    self.diffs_norm = 0
    self.recv_thread = None
    self.result_buffer = None

  def update(self, datas):
    self.error = 0
    self.result.clear()
    self.result_buffer = None
    diffs = []
    for buffer, data in zip(self.buffers, datas):
      data = data.to(self.device)
      diffs.append(data.sub(buffer))
    radius = 0
    temp = torch.zeros(1).to(self.device)
    for diff in diffs:
      torch.norm(input = diff, p = float('inf'), out = temp)
      radius = max(temp.item(), radius)
    self.scale = radius*2/((1<<self.data_width)-1)
    self.diffs_norm = 0
    for diff in diffs:
      result_tensor = diff.add(radius).div_(self.scale).add_(0.5).long()
      self.result.extend(result_tensor.flatten().tolist())
      qd = result_tensor.float().mul_(self.scale).sub_(radius)
      self.error += torch.norm(diff.sub_(qd), p = 2).pow_(2).item()
      diff.copy_(qd)
      torch.norm(diff, p = 2, out = temp)
      self.diffs_norm += temp.pow_(2).item()

  def step(self):
    result = self.result
    radius = ((1<<self.data_width)-1)*self.scale/2
    for buffer in self.buffers:
      num_buffer = buffer.size().numel()
      diff, result = result[:num_buffer], result[num_buffer:]
      diff = torch.tensor(diff, device = self.device).float().resize_(buffer.size()).mul_(self.scale)
      buffer.add_(diff).sub_(radius)

  def send_to(self, dst):
    if self.result_buffer == None:
      self.result_buffer = pack_segs(self.result, self.data_width)
    size = len(self.result_buffer)+getsizeof(self.scale)
    dist.send(torch.tensor([self.scale]), dst)
    send_bytes(self.result_buffer, dst)
    return size

  def recv_from(self, src, with_buffers = False):
    recv_conn, self.scale, recv_len, self.recv_proc = recv_segs(src, self.data_width)
    self.result.clear()
    self.result_buffer = None
    def recv_pipe():
      for seg in iter(recv_conn.recv, EOP):
        self.result.append(seg)
      recv_conn.close()
    self.recv_thread = Thread(target=recv_pipe)
    self.recv_thread.start()
    return recv_len
  
  def wait_recv(self):
    if self.recv_thread != None:
      self.recv_thread.join()
      self.recv_thread = None
  
  def reset_buffer(self):
    for buffer in self.buffers:
      buffer.fill_(0)

class QGD(Optimizer):
  def __init__(self, params, qbuf, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, qbuf=qbuf)
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    super(QGD, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(QGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']
      qbuf = group['qbuf']
      diffs = []
      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad
        if weight_decay != 0:
          d_p = d_p.add(p, alpha=weight_decay)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          if nesterov:
            d_p = d_p.add(buf, alpha=momentum)
          else:
            d_p = buf
        diffs.append(d_p)
      qbuf.update(diffs)
    return loss