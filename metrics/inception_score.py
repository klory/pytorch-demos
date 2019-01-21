# original paper: https://arxiv.org/abs/1606.03498
# orignal code [tf]: https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# mostly copy from [pytorch]: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

def inception_score(imgs, batch_size=128, splits=10):
    """Computes the inception score of the images
    imgs -- Torch dataset of [N, 3, H ,W] images normalized in the range [-1, 1], NO LABELS!
    batch_size -- batch size for feeding into Inception v3
    """
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=> Device is', device)

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    if torch.cuda.device_count() > 1:
        print("=> Let's use {} GPUs".format(torch.cuda.device_count()))
        inception_model = nn.DataParallel(inception_model)
    inception_model.eval()

    # Get predictions
    print('=> Get predictions...')
    preds = []
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        x = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=True)
        x = inception_model(x)
        preds.append(F.softmax(x, dim=1).detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Now compute the mean kl-div
    print('=> Compute KL[p(y) | p(y|x)]...')
    split_scores = []
    for k in range(splits):
        bs = N // splits
        part = preds[k * bs: (k+1) * bs, :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyGx = part[i, :]
            scores.append(entropy(pyGx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='./cifar10', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize([32,32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )
    
    # only use the first 20 images for demonstration
    cifar = torch.utils.data.Subset(cifar, range(20))
    IgnoreLabelDataset(cifar)
    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), batch_size=2, splits=10))