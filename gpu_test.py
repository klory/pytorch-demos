import torch
from torchvision.models import resnet50
import pdb
import os
import argparse

parser = argparse.ArgumentParser(description='retrieval model parameters')
parser.add_argument('gpus', default='0', type=str)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

device = torch.device('cuda')
m = torch.nn.DataParallel(resnet50()).to(device)
gpus = torch.cuda.device_count()
batch_size = 96
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters())
epoch = 0
while 1:
    x = torch.randn(batch_size*gpus, 3, 224, 224).to(device)
    target = (torch.rand(batch_size*gpus)*1000).long().to(device)
    y = m(x)
    loss = criterion(y, target)
    print('[{:>8d}] loss = {:.4f}'.format(epoch, loss))
    epoch += 1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
