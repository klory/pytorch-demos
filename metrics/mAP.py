import torch
from torch import nn, optim
from torch.nn import functional as F
from torchnet import meter

# here we directly use torchnet's implementation, so we do not need \
# recreate the wheel, they have plenty of other metrics for \
# classification and regression, check there website for more.

# both output and target are [N,K], here N means we have N samples, \
# K means we have K classes for each sample

# torchnet meter does NOT require the model output to be between [0, 1], \
# here we have N=2 and K=4
output = torch.tensor([
    [1.8, 3.3, 0.6, 4.02], 
    [0.6, 5.22, 1.45, 0.95]]).float()

# the target has the same shape as the output, all values are either 0 \
# or 1. Here, sample_1 has one class, sample_2 has three classes
target = torch.tensor([
    [0, 1, 0, 0], 
    [0, 1, 1, 1]]).float()


mtr = meter.APMeter()
mtr.add(output, target)
# call mtr.value() will return the APs (average precision) for each \
# class, so the output will have shape [K]
print(mtr.value())
print(mtr.value().mean()) # then this is mAP