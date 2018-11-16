import torch
from torch import nn, optim
from torch.nn import functional as F

x = torch.randn(2, 100, requires_grad=True)
model = nn.Sequential(
    nn.Linear(100,50),
    nn.ReLU(),
    nn.Linear(50, 4)
)
optimizer = optim.Adam(model.parameters())

target = torch.FloatTensor([[0,1,1,0], [0,0,1,1]])

criterion = nn.MultiLabelSoftMarginLoss()

from torchnet import meter


for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print('\nepoch = {}, loss = {:.4f}'.format(epoch, loss))
    print(F.sigmoid(output))

    mtr = meter.APMeter()
    mtr.add(output.t(), target.t())
    print(mtr.value())
    print(mtr.value().mean())