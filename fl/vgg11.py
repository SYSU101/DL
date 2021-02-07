import torch
import torch.nn as nn
from torchvision.models import vgg11
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class VGG11(nn.Module):
  def __init__(self, **kwargs):
    super(VGG11, self).__init__()
    origin = vgg11(**kwargs)
    self._features = origin.features
    self.avgpool = origin.avgpool
    self._classifier = origin.classifier
  
  def features(self, x):
    x = checkpoint_sequential(self._features, 4, x)
    return x
  
  def classifier(self, x):
    x = checkpoint_sequential(self._classifier, 2, x)
    return x
  
  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
