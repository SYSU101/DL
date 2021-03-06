import torch
from .communication import send_segs, recv_segs

class QuantizedBuffer(object):
  def __init__(self, data_len, data_width, device):
    self.data_len = data_len
    self.buffers = [None for _ in range(data_len)]
    self.diffs = [None for _ in range(data_len)]
    self.device = device
    self.data_width = data_width
    self.scale = 0
    self.result = []
    self.error = 0

  def update(self, datas):
    self.error = 0
    self.result.clear()
    indexes = range(self.data_len)
    for index, data in zip(indexes, datas):
      data = data.to(self.device)
      buffer = self.buffers[index]
      if buffer == None:
        self.diffs[index] = data.clone().detach()
        self.buffers[index] = data.clone().detach()
      else:
        self.diffs[index] = data.sub(buffer)
    radius = 0
    temp = torch.zeros(1)
    for diff in self.diffs:
      torch.norm(input = grad_diff, p = float('inf'), out = temp)
      radius = max(temp.item(), radius)
    self.scale = r*2/self.data_width
    self.calc_diffs()
  
  def calc_diffs(self):
    radius = self.scale*self.data_width/2
    def quantize_diff(diff):
      result = ((diff+radius)/self.scale+0.5)//1
      self.result.append(c_uint32(int(result)))
      result = result*self.scale-radius
      self.error += abs(result-diff)
      return result
    for diff in self.diffs:
      diff.apply_(quantize_diff)

  def step(self, datas, alpha = 1.0):
    for data, diff in zip(datas, self.diffs):
      data.to(self.device).add_(diff, alpha)
  
  def diffs_norm(self):
    temp = torch.zeros(1)
    result = 0
    for diff in self.diffs:
      torch.norm(diff, p = 2, out = temp)
      result += temp.pow_(2).item()
    return result

  def send_to(self, dst):
    return send_segs(self.result, self.scale, self.data_width, dst)

  def recv_from(self, src):
    self.result, self.scale, recv_len = recv_segs(src, self.data_width)
    return recv_len
