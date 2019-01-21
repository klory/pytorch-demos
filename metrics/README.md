# IS-FID-PyTorch
A SIMPLE Inception Score and Frechet Inception Distance implementation with PyTorch 1.0.0

The code supports multi-GPU by running (the code will **use ALL visible devices**, so be sure to specify visible GPUs using `CUDA_VISIBLE_DEVICES`):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py
```

## Environment
Only tested in Python 3.6 on Macbook Pro 2012

## Requirements
see [requirements.txt](https://github.com/klory/IS-FID-PyTorch/blob/master/requirements.txt)

## Inception Score (IS)
The table shows the results I got from CIFAR10 (50k)

| number of splits | mean           | std  |
| -------------    |:-------------: | -----:|
| 5                | 9.3942         | 0.0838
| 10               | 9.3701         | 0.1496
| 20               | 9.3215         | 0.1900

In the orignal paper, they suggest 

> a large enough number of samples (i.e. 50k)

and on [this line](https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L24) of their original code, they do it by 10 splits. So I assume they divide the original CIFAR10 (50k) images by 10 splits, which means each split has 5k images.

The weights of [Inception V3 in tensorflow](https://github.com/openai/improved-gan/blob/master/inception_score/model.py) is different from [That in PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py). So I could not get exactly same scores on CIFAR10 in the [original Inception Score paper](https://arxiv.org/abs/1606.03498). I got `9.3701, 0.1496` instead of `11.24, 0.12` (much lower, sigh...)

Below is the result from the original paper:

![CIFAR10 Inception Score from orignal paper](https://github.com/klory/IS-FID-PyTorch/blob/master/images/orignal-IS-CIFAR10.png)

One **Problem** I found with the orignal IS implementation in it only support `batch_size=1`, which is not quite pleasing:
- https://github.com/tensorflow/tensorflow/issues/554
- https://github.com/jhjin/tensorflow-cpp/issues/1
- https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

### reference
- original paper: https://arxiv.org/abs/1606.03498
- orignal code [tf]: https://github.com/openai/improved-gan/blob/master/inception_score/model.py
- mostly copy from [pytorch]: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

## Frechet Inception Distance (FID)
FID needs to estimate the mean and variance from image features, each feature is 2048-d, so we would have a 2048x2048 covariance matrix, in order to estimate the cov more accurate, I would suggest to make the number of images at least ten times larger than the dimension, say 20480.

FID needs to calculate the [square root of the matrix](https://en.wikipedia.org/wiki/Square_root_of_a_matrix)
```
covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
```
`covmean` might be close to singular, so we need to take care of numerical issues.

### reference
- original paper: https://arxiv.org/abs/1706.08500
- orignal code [tf]: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
- mostly copy from [pytorch]: https://github.com/mseitzer/pytorch-fid
