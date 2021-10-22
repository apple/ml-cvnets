# CVNets: A library for training computer vision networks

This repository contains the source code for training computer vision models. Specifically, it contains the source code of the [MobileViT](https://arxiv.org/abs/2110.02178?context=cs.LG) paper for the following tasks:
   * Image classification on the ImageNet dataset
   * Object detection using [SSD](https://arxiv.org/abs/1512.02325)
   * Semantic segmentation using [Deeplabv3](https://arxiv.org/abs/1706.05587)

***Note***: Any image classification backbone can be used with object detection and semantic segmentation models

Training can be done with two samplers:
   * Standard distributed sampler
   * [Mulit-scale distributed sampler](https://arxiv.org/abs/2110.02178?context=cs.LG)

We recommend to use multi-scale sampler as it improves generalization capability and leads to better performance. See [MobileViT](https://arxiv.org/abs/2110.02178?context=cs.LG) for details.

## Installation

CVNets can be installed in the local python environment using the below command:
``` 
    git clone git@github.com:apple/ml-cvnets.git
    cd ml-cvnets
    pip install -r requirements.txt
    pip install --editable .
```

We recommend to use Python 3.6+ and [PyTorch](https://pytorch.org) (version >= v1.8.0) with `conda` environment. For setting-up python environment with conda, see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Getting Started

   * General instructions for training and evaluation different models are given [here](README-training-and-evaluation.md). 
   * Examples for a training and evaluating a specific model are provided in the [examples](examples) folder. Right now, we support following models.
     * [MobileViT](examples/README-mobilevit.md) 
     * [MobileNetv2](examples/README-mobilenetv2.md) 
     * [ResNet](examples/README-resnet.md)
   * For converting PyTorch models to CoreML, see [README-pytorch-to-coreml.md](README-pytorch-to-coreml.md).

## Citation

If you find our work useful, please cite the following paper:

``` 
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```
