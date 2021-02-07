#! /usr/local/bin python3.8

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from . import config

def init_process(rank, world_size, fn, args):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group('gloo', rank=rank, world_size=world_size)
  fn(rank, world_size, *args)

def simulate(server_fn, server_args, client_fn, client_num = config.client_num, gen_client_args):
  world_size = client_num+1
  processes = [
    Process(target=init_process, args=(0, world_size, serverfn, server_args))
  ]
  for rank in range(1, client_num+1):
    client_args = gen_client_args(rank)
    processes.append(
      Process(target=init_process, args=(rank, world_size, client_fn, client_args))
    )
  for p in processes:
    p.start()
  for p in processes:
    p.join()
