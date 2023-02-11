# Deep Convolutional Architectures

## Resources

- [Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- [1x1 Convolutions](https://youtu.be/SIpcirNNGAk)
- [What does 1x1 convolution mean in a neural network?](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network)
- [GoogLeNet Tutorial](https://youtu.be/_XF7N6rp9Jw)
- [Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image Classification)](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7)
- [Residual Neural Network](https://en.wikipedia.org/wiki/Residual_neural_network)
- [Review: ResNet — Winner of ILSVRC 2015 (Image Classification, Localization, Detection)](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8)
- [Deep Residual Learning for Image Recognition](https://youtu.be/C6tLw-rPQ2o)
- [Review: ResNeXt — 1st Runner Up in ILSVRC 2016 (Image Classification)](https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac)
- [Review: DenseNet — Dense Convolutional Network (Image Classification)](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)
- [Densely Connected Convolutional Networks](https://youtu.be/-W6y8xnd--U)
- [Network In Network](https://youtu.be/c1RBQzKsDCk)
- [Inception Network Motivation](https://youtu.be/C86ZXvgpejM)
- [Inception Network](https://youtu.be/KfV8CJh7hE0)
- [Resnets](https://youtu.be/ZILIbUvp5lk)
- [Why ResNets Work](https://youtu.be/RYth6EbBUqM)
- [Network in Network (2014)](https://arxiv.org/pdf/1312.4400.pdf)
- [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf)
- [Highway Networks (2015)](https://arxiv.org/pdf/1505.00387.pdf)
- [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf)
- [Aggregated Residual Transformations for Deep Neural Networks (2017)](https://arxiv.org/pdf/1611.05431.pdf)
- [Densely Connected Convolutional Networks (2018)](https://arxiv.org/pdf/1608.06993.pdf)
- [Multi-Scale Dense Networks for Resource Efficient Image Classification (2018)](https://arxiv.org/pdf/1703.09844.pdf)

## Learning Objectives

- What is a skip connection?
- What is a bottleneck layer?
- What is the Inception Network?
- What is ResNet? ResNeXt? DenseNet?
- How to replicate a network architecture by reading a journal article

## Tasks

| Filename                 | Description                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| `0-inception_block.py`   | Builds an inception block as described in Going Deeper with Convolutions (2014)                       |
| `1-inception_network.py` | Builds the inception network as described in Going Deeper with Convolutions (2014)                    |
| `2-identity_block.py`    | Builds an identity block as described in Deep Residual Learning for Image Recognition (2015)          |
| `3-residual_block.py`    | Builds a residual block as described in Deep Residual Learning for Image Recognition (2015)           |
| `4-resnet50.py`          | Builds the ResNet-50 architecture as described in Deep Residual Learning for Image Recognition (2015) |
| `5-dense_block.py`       | Builds a dense block as described in Densely Connected Convolutional Networks (2017)                  |
| `6-transition_layer.py`  | Builds a transition layer as described in Densely Connected Convolutional Networks (2017)             |
| `7-densenet121.py`       | Builds the DenseNet-121 architecture as described in Densely Connected Convolutional Networks (2017)  |
