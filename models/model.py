import torch
import torch.nn as nn
import torch.nn.functional as F

# from .vit import VIT
from . import model_utils as mu


class Model(nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()
    self.config = config

    # Define backbone (currently only resnet useful):    
    if self.config.model_config.use_backbone:
      root_token_dim = 64 
      self.conv_root = mu.Conv2d(3, root_token_dim, (7, 7), (2,2), padding=0, bias=True)
      self.gn_root = nn.GroupNorm(32, root_token_dim)
      self.relu_root = nn.ReLU()
      self.maxpool_root = nn.MaxPool2d((3,3), (2,2))
      self.res_blocks = None

    # Define embedding layer to get embedding size match with Transformers:
    in_embedding = 256*8*8 if self.config.model_config.use_backbone else 32*32*3
    self.embedding = nn.Linear(in_embedding, self.config.model_config.hidden_size)

    # Define VIT:
    # self.VIT = VIT(self.config)

    # Define head:
    self.head = nn.Linear(self.config.model_config.hidden_size, self.config.model_config.n_classes)

    # Initialize all weights:
    self.apply(mu._weight_init)

  def forward(self, x):
    B, T, channel = x.shape 
    
    if self.config.model_config.hidden_size:
      x = x.reshape(-1, self.config.data_config.patch_size, self.config.data_config.patch_size, 3)
      x = x.permute(0, 3, 1, 2)
      # ToDO: Modify some of the padding as it was only done because of tf.
      x = F.pad(x, (0, 1, 0, 1))
      x = F.pad(self.relu_root(self.gn_root(self.conv_root(x))), (0,1,0,1))
      x = self.maxpool_root(x)
      x = self.res_blocks(x)
      x = x.permute(0, 2, 3, 1)
      x = x.reshape([B, T, -1])

    x = self.embedding(x)
    # x = self.VIT(x)
    x = self.head(x[:, 0])
    return x

