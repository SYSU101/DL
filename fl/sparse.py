from heapq import *
from torch.optim.optimizer import Optimizer, required
from struct import pack, unpack

class Sparser(object):
  def __init__(self, p, tot_num):
    self.k = tot_num*p
    self.heap = []
    self.buffers = []
    self.valid = []
  
  def push(grads):
    grads = grads.flatten().abs_().tolist()
    for grad in grads:
      valid = False
      if len(self.heap) < self.k:
        heappush(self.heap, grad)
        valid = True
      elif grad >= self.heap[0]:
        heappushpop(self.heap, grad)
        valid = True
      if valid:
        self.buffers.append(grad)
  
  def reset():
    self.heap.clear()
    self.outputs.clear()
    self.valid.clear()

  def output():
    bi = 0
    output = []
    threshold = self.heap[0]
    for vi in range(len(self.valid)):
      valid = self.valid[vi]
      if valid:
        buffer = self.buffers[bi]
        bi += 1
        if buffer >= threshold:
          output.append(buffer)
        else:
          self.valid[vi] = False
    return output, self.valid.copy()

class SparseSGD(Optimizer):
  def __init__(self, params, sparser, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(
      lr=lr,
      momentum=momentum,
      dampening=dampening,
      weight_decay=weight_decay,
      nesterov=nesterov,
      sparser=sparser
    )
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    super(SparseSGD, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(SparseSGD, self).__setstate__(state)
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
      sparser = group['sparser']
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
        sparser.push(d_p)
    return loss

def apply_sparse_grads(params, grads, valid, lr):
  grads = iter(grads)
  for param in params:
    for i in range(param.numel()):
      if valid[i]:
        get_el_by_offset(param, i).add_(next(grads), alpha=lr)
    valid = valid[param.numel():]

def get_el_by_offset(tensor, offset):
  sizes = tensor.size()
  numel = tensor.numel()
  dim = tensor.dim()
  for i in range(dim):
    numel = int(numel/sizes[i])
    index = int(offset/numel)
    offset = offset%numel
    tensor = tensor[index]
  return tensor
