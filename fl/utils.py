import torch
import sys
from sys import stdout
from itertools import chain
from torch.utils.data import DataLoader

def test_accuracy(model, testset, device):
  tests = DataLoader(testset, batch_size=1, shuffle=True)
  total = 0
  correct = 0
  for data in tests:
    image, label = data
    outputs = model(image.to(device))
    predicted = torch.argmax(outputs.data, 1)
    total += 1
    correct += (predicted == label.to(device)).sum().item()
    if total > 100:
      break
  return correct/total

def debug_print(*args, **kwargs):
  print(*args, **kwargs)
  stdout.flush()

def save_lists(path, *lists):
  f = open(path, 'w')
  for l in lists:
    f.write(','.join(map(str, l)))
    f.write('\n')
  f.close()

def read_lists(path):
  lists = []
  f = open(path, 'r')
  lines = f.readlines()
  for line in lines:
    lists.append(list(map(float, line.split(','))))
  return lists

def once(fn):
  tag = True
  def do_once(*args, **kwargs):
    nonlocal tag
    if tag:
      fn(*args, **kwargs)
      tag = False
  return do_once

def decay_learning_rate(optimizer, alpha, min_lr):
  for param_group in optimizer.param_groups:
    new_lr = param_group['lr']*alpha
    param_group['lr'] = max(new_lr, min_lr)

def clear_params(params, buffer = None):
  for param in params:
    param.data.fill_(0)
    if param.data.grad != None:
      param.grad.fill_(0)
  if buffer != None:
    for buf in buffer:
      buf.data.fill_(0)

def get_params(model):
  params = model.parameters()
  if model.buffers() != None:
    params = chain(params, model.buffers())
  return params

def get_grads(model):
  params = model.parameters()
  params = map(lambda p: p.grad, params)
  if model.buffers() != None:
    params = chain(params, model.buffers())
  return params