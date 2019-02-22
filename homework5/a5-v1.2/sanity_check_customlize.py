
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN

in_channel= 5
out_channel = 2
k = 5
hw = CNN(size1, size2)

assert hw.cov1d_layer.in_channels == in_channel
assert hw.cov1d_layer.out_channels == out_channel
assert hw.cov1d_layer.kernel_size == k

t = torch.randn((5,5))
out = hw(t)
assert t.size() == out.size()


