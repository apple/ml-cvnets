# Training ResNets on the ImageNet dataset

Single node 8-GPU training of `ResNet-50` with `simple training recipe` can be done using below command:

``` 
export CFG_FILE="config/classification/imagenet/resnet.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

For advanced training recipe, see [this](../../../../../config/classification/imagenet/resnet_adv.yaml) configuration file.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

<details>
<summary>
Single node 8-GPU training of ResNet-101 with simple training recipe
</summary>

``` 
export CFG_FILE="config/classification/imagenet/resnet.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --common.override-kwargs model.classification.resnet.depth=101
```
</details>


<details>
<summary>
Single node 8-GPU training of ResNet-34 with simple training recipe
</summary>

``` 
export CFG_FILE="config/classification/imagenet/resnet.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --common.override-kwargs model.classification.resnet.depth=34
```
</details>

## Citation

``` 
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
