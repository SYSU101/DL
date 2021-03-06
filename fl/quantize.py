import torch
from ctypes import c_uint32
from .communication import send_segs, recv_segs, EOP
from .utils import debug_print

class QuantizedBuffer(object):
  def __init__(self, data_len, data_width, device):
    self.data_len = data_len
    self.buffers = [None for _ in range(data_len)]
    self.device = device
    self.data_width = data_width
    self.scale = 0
    self.result = []
    self.error = 0
    self.diffs_norm = 0
    self.recv_proc = None
    self.recv_conn = None

  def update(self, datas):
    self.error = 0
    self.result.clear()
    indexes = range(self.data_len)
    diffs = []
    for index, data in zip(indexes, datas):
      data = data.to(self.device)
      buffer = self.buffers[index]
      if buffer == None:
        diffs.append(data.clone().detach())
        self.buffers[index] = data.clone().detach()
      else:
        diffs.append(data.sub(buffer))
    radius = 0
    temp = torch.zeros(1).to(self.device)
    for diff in diffs:
      torch.norm(input = diff, p = float('inf'), out = temp)
      radius = max(temp.item(), radius)
    self.scale = radius*2/self.data_width
    def quantize_diff(diff):
      result = ((diff+radius)/self.scale+0.5)//1
      self.result.append(c_uint32(int(result)))
      result = result*self.scale-radius
      self.error += (result-diff)**2
      return result
    for diff in diffs:
      diff.to(torch.device('cpu')).apply_(quantize_diff)
    temp.fill_(0)
    self.diffs_norm = 0
    for diff in diffs:
      torch.norm(diff, p = 2, out = temp)
      self.diffs_norm += temp.pow_(2).item()

  def step(self, datas, alpha = 1.0):
    result = iter(self.result)
    radius = self.data_width*self.scale/2
    for data in datas:
      data.to(torch.device('cpu')).apply_(lambda d: d+(next(result)*self.scale-radius)*alpha)

  def send_to(self, dst):
    return send_segs(self.result, self.scale, self.data_width, dst)

  def recv_from(self, src):
    self.recv_conn, self.scale, recv_len, self.recv_proc = recv_segs(src, self.data_width)
    return recv_len
  
  def wait_recv(self):
    if self.recv_proc != None:
      self.result.clear()
      for seg in iter(self.recv_conn.recv, EOP):
        self.result.append(seg)
      self.recv_conn.close()
      self.recv_proc.join()
      self.recv_proc = None
      self.recv_conn = None
