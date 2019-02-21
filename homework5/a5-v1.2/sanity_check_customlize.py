import torch
from cnn import CNN
from highway import Highway

cnn = CNN(1,1,3)
cnn.cov1d_layer.weight.data.fill_(1)
cnn.cov1d_layer.bias.data.fill_(0)
input = torch.Tensor(2,1,5)
#input[0] = torch.Tensor([1,2,3,4,5])
#input[1] = torch.Tensor([6,7,8,9,10])

input[0] = torch.Tensor([1,2,3,-4,-5])
input[1] = torch.Tensor([6,7,8,-9,-10])
out = cnn(input)

print(out)
print('*' * 10)

h = Highway(2)
h.gate_layer.bias.data.fill_(0)
h.proj_layer.bias.data.fill_(0)

h.proj_layer.weight.data.fill_(1)
h.gate_layer.weight.data.fill_(1)

input = torch.Tensor(2,2)
input[0] = torch.Tensor([0.1,0.2])
input[1] = torch.Tensor([0.6,0.7])

out = h(input)
print(out)