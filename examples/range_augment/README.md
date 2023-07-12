# RangeAugment: Efficient Online Augmentation with Range Learning

[RangeAugment](https://arxiv.org/abs/2212.10553) is an automatic augmentation method that allows us to learn `model- and task-specific` magnitude range of each augmentation operation.

We provide training and evaluation code along with pretrained models and configuration files for the following tasks:

1. [Image Classification on the ImageNet dataset](./README-classification.md)
2. [Semantic segmentation on the ADE20k and the PASCAL VOC datasets](./README-segmentation.md)
3. [Object detection on the MS-COCO dataset](./README-object-detection.md)
4. [Contrastive Learning using Image-Text pairs](./README-clip.md)
5. [Distillation on the ImageNet dataset](./README-distillation.md)

***Note***: In the [codebase](../../cvnets/neural_augmentor), we refer RangeAugment as Neural Augmentor (or NA).


## Citation

If you find our work useful, please cite:

``` 
@article{mehta2022rangeaugment,
  title={RangeAugment: Efficient Online Augmentation with Range Learning},
  author = {Mehta, Sachin and Naderiparizi, Saeid and Faghri, Fartash and Horton, Maxwell and Chen, Lailin and Farhadi, Ali and Tuzel, Oncel and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2212.10553},
  year={2022},
  url={https://arxiv.org/abs/2212.10553},
}
```
