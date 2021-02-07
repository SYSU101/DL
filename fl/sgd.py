#! /usr/bin/env python3.8

from torch.optim.optimizer import Optimizer, required

'''
  This struct generally follows common SGD optimizer of pytorch.
  Updates will be stored in .grad instead of direct update on parameters,
  so that some other optimizers could be applied.
'''
class FedSGD(Optimizer):
  def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    '''Implements stochastic gradient descent (optionally with momentum).

      Args:
        params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    '''
    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    super(SGD, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(SGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)

  def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']

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
        d_p.mul_(-group['lr'])
        p.grad.copy_(d_p)

    return loss