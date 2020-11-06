import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models
import numpy as np

class INCEPTION_V3(nn.Module):
    """
    images is between [-1, 1]
    """
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained inception_v3 model from', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.model(x)
        # import pdb; pdb.set_trace()
        x = nn.Softmax(dim=-1)(x)
        return x


def compute_is(predictions, num_splits=1):
    """Computes Inception Score (IS) for image predictions
    predictions -- Torch dataset of [Nr, 1000]: outputs of Inception_v3 network
    num_splits -- how many times to compute IS.
    """
    predictions = predictions.cpu().numpy()
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

if __name__ == '__main__':
    model = INCEPTION_V3()
    img = torch.tanh(torch.randn(32, 3, 256, 256))
    with torch.no_grad():
        predictions = model(img)
    print(compute_is(predictions, num_splits=1))
