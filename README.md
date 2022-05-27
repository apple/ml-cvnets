# CVNets: A library for training computer vision networks

This repository contains the source code for training computer vision models for different tasks, including ImageNet-1k/21k classification,
MS-COCO object detection, ADE20k semantic segmentation, and Kinetics-400 video classification.

## Installation

CVNets can be installed in the local python environment using the below command:
``` 
    git clone git@github.com:apple/ml-cvnets.git
    cd ml-cvnets
    pip install -r requirements.txt
    pip install --editable .
```

We recommend to use Python 3.7+ and [PyTorch](https://pytorch.org) (version >= v1.8.0) with `conda` environment. For setting-up python environment with conda, see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Getting Started

   * General instructions for working with CVNets are given [here](docs/source/en/general).
   * Examples for training and evaluating models are provided [here](docs/source/en/models).
   * Examples for converting a PyTorch model to CoreML are provided [here](docs/source/en/general/README-pytorch-to-coreml.md).

## Model Zoo

For benchmarking results including novel and existing models, see [Model Zoo](docs/source/en/general/README-model-zoo.md). 

## The Team

CVNets is currently maintained by <a href="https://sacmehta.github.io" target="_blank">Sachin Mehta</a> and <a href="https://farzadab.github.io" target="_blank">Farzad Abdolhosseini</a>.

## Contributing

We welcome PRs from the community! You can find information about contributing to CVNets in our [contributing](CONTRIBUTING.md) document. 

Please remember to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

For license details, see [LICENSE](LICENSE). 

## Citation

If you find CVNets useful, please cite following papers:

``` 
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}

@article{mehta2022cvnets,
    title={CVNets: High Performance Library for Computer Vision},
    author={Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
    year={2022}
}
```