# CVNets: A library for training computer vision networks

CVNets is a computer vision toolkit that allows researchers and engineers to train standard and novel mobile- 
and non-mobile computer vision models for variety of tasks, including object classification, object detection,
semantic segmentation, and foundation models (e.g., CLIP).

## Table of contents

   * [What's new?](#whats-new)
   * [Installation](#installation)
   * [Getting started](#getting-started)
   * [Supported models and tasks](#supported-models-and-tasks)
   * [Maintainers](#maintainers)
   * [Research effort at Apple using CVNets](#research-effort-at-apple-using-cvnets)
   * [Contributing to CVNets](#contributing-to-cvnets)
   * [License](#license)
   * [Citation](#citation)

## What's new?

   * ***July 2023***: Version 0.4 of the CVNets library includes
      *  [Bytes Are All You Need: Transformers Operating Directly On File Bytes
](https://arxiv.org/abs/2306.00238)
      * [RangeAugment: Efficient online augmentation with Range Learning](https://arxiv.org/abs/2212.10553)
      * Training and evaluating foundation models (CLIP)
      * Mask R-CNN
      * EfficientNet, Swin Transformer, and ViT
      * Enhanced distillation support

## Installation

We recommend to use Python 3.10+ and [PyTorch](https://pytorch.org) (version >= v1.12.0)

Instructions below use Conda, if you don't have Conda installed, you can check out [How to Install Conda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).

```bash
# Clone the repo
git clone git@github.com:apple/ml-cvnets.git
cd ml-cvnets

# Create a virtual env. We use Conda
conda create -n cvnets python=3.10.8
conda activate cvnets

# install requirements and CVNets package
pip install -r requirements.txt -c constraints.txt
pip install --editable .
```

## Getting started

   * General instructions for working with CVNets are given [here](docs/source/en/general). 
   * Examples for training and evaluating models are provided [here](docs/source/en/models) and [here](examples). 
   * Examples for converting a PyTorch model to CoreML are provided [here](docs/source/en/general/README-pytorch-to-coreml.md).

## Supported models and Tasks

To see a list of available models and benchmarks, please refer to [Model Zoo](docs/source/en/general/README-model-zoo.md) and [examples](examples) folder.

<details>
<summary>
ImageNet classification models
</summary>

   * CNNs
     * [MobileNetv1](https://arxiv.org/abs/1704.04861)
     * [MobileNetv2](https://arxiv.org/abs/1801.04381)
     * [MobileNetv3](https://arxiv.org/abs/1905.02244)
     * [EfficientNet](https://arxiv.org/abs/1905.11946)
     * [ResNet](https://arxiv.org/abs/1512.03385)
     * [RegNet](https://arxiv.org/abs/2003.13678)
   * Transformers
     * [Vision Transformer](https://arxiv.org/abs/2010.11929)
     * [MobileViTv1](https://arxiv.org/abs/2110.02178)
     * [MobileViTv2](https://arxiv.org/abs/2206.02680)
     * [SwinTransformer](https://arxiv.org/abs/2103.14030)
</details>

<details>
<summary>
Multimodal Classification
</summary>

  * [ByteFormer](https://arxiv.org/abs/2306.00238)

</details>

<details>
<summary>
Object detection
</summary>

   * [SSD](https://arxiv.org/abs/1512.02325)
   * [Mask R-CNN](https://arxiv.org/abs/1703.06870)

</details>

<details>
<summary>
Semantic segmentation
</summary>

   * [DeepLabv3](https://arxiv.org/abs/1706.05587)
   * [PSPNet](https://arxiv.org/abs/1612.01105)

</details>

<details>
<summary>
Foundation models
</summary>

   * [CLIP](https://arxiv.org/abs/2103.00020)

</details>

<details>
<summary>
Automatic Data Augmentation
</summary>

   * [RangeAugment](https://arxiv.org/abs/2212.10553)
   * [AutoAugment](https://arxiv.org/abs/1805.09501)
   * [RandAugment](https://arxiv.org/abs/1909.13719)

</details>

<details>
<summary>
Distillation
</summary>

   * Soft distillation
   * Hard distillation

</details>

## Maintainers

This code is developed by <a href="https://sacmehta.github.io" target="_blank">Sachin</a>, and is now maintained by Sachin, <a href="https://mchorton.com" target="_blank">Maxwell Horton</a>, <a href="https://www.mohammad.pro" target="_blank">Mohammad Sekhavat</a>, and Yanzi Jin.

### Previous Maintainers
* <a href="https://farzadab.github.io" target="_blank">Farzad</a>

## Research effort at Apple using CVNets

Below is the list of publications from Apple that uses CVNets:

   * [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer, ICLR'22](https://arxiv.org/abs/2110.02178)
   * [CVNets: High performance library for Computer Vision, ACM MM'22](https://arxiv.org/abs/2206.02002)
   * [Separable Self-attention for Mobile Vision Transformers (MobileViTv2)](https://arxiv.org/abs/2206.02680)
   * [RangeAugment: Efficient Online Augmentation with Range Learning](https://arxiv.org/abs/2212.10553)
   * [Bytes Are All You Need: Transformers Operating Directly on File Bytes](https://arxiv.org/abs/2306.00238)

## Contributing to CVNets

We welcome PRs from the community! You can find information about contributing to CVNets in our [contributing](CONTRIBUTING.md) document. 

Please remember to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

For license details, see [LICENSE](LICENSE). 

## Citation

If you find our work useful, please cite the following paper:

``` 
@inproceedings{mehta2022mobilevit,
     title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
     author={Sachin Mehta and Mohammad Rastegari},
     booktitle={International Conference on Learning Representations},
     year={2022}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}

```
