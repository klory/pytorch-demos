from torch import nn
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.fc = nn.Linear(num_classes, num_features * 2)
    self.fc.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.fc.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, img, label):
    out = self.bn(img)
    gamma, beta = self.fc(label).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out