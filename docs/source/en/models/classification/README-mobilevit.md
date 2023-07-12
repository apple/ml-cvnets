# Training MobileViT on the ImageNet dataset

Single node 8-GPU training of `MobileViT-S` can be done using below command:

``` 
export CFG_FILE="config/classification/imagenet/mobilevit.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Citation

``` 
@inproceedings{mehta2022mobilevit,
    title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
    author={Sachin Mehta and Mohammad Rastegari},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=vh-0sUt8HlG}
}
```
