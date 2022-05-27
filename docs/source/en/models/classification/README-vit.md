# Training Classification Models on the ImageNet dataset

## Training Visual Transformer on the ImageNet dataset

Single node 4-GPU training of `ViT-Tiny` can be done using below command:

``` 
export CFG_FILE="config/classification/imagenet/vit.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Citation

``` 
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

@inproceedings{touvron2021training,
  title={Training data-efficient image transformers \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  pages={10347--10357},
  year={2021},
  organization={PMLR}
}
```