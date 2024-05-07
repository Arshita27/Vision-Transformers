import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Conv2d(nn.Conv2d):
  """
  Refer to: https://github.com/joe-siyuan-qiao/WeightStandardization
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias)

  def forward(self, x, apply_weight_standardization):
      if apply_weight_standardization:
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
      else:
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def _weight_init(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    init.kaiming_normal_(m.weight)