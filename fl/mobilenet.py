from torch.nn import BatchNorm2d
import torch
import torchvision.models as models

def norm_layer(features):
  base = BatchNorm2d(features)
  base.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.float64))
  return base

def mobilenet_v2(*args, **kwargs):
  return models.mobilenet_v2(*args, norm_layer = norm_layer, **kwargs)