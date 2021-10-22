# Training ResNet-50 with CVNets on the ImageNet-1k dataset

For a machine with 8 NVIDIA A100 GPUs, ResNet-50 model can be trained by running the following command from the root directory:
``` 
cvnets-train --common.config-file config/classification/resnet.yaml --common.results-loc results_imagenet1k_resnet50
```
Note that the default location of the ImageNet-1k training and validation sets in the configuration file are `/mnt/imagenet/training` and `/mnt/imagenet/validation` respectively. Please make changes accordingly in the [configuration file](../config/classification/resnet.yaml).

Evaluation on the ImageNet-1k dataset can be achieved using `cvnets-eval` command. Let us assume that ResNet-50 model weight file name is `checkpoint_ema.pt` and is stored in `results_imagenet1k_resnet50` folder. Also, this folder should also contain `config.yaml` file, which is nothing but a copy of the configuration file that was used during training. 

Now, we can run the following command to evaluate the performance on GPU-0:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file results_imagenet1k_resnet50/config.yaml --common.results-loc results_imagenet1k_resnet50 --model.classification.pretrained results_imagenet1k_resnet50/checkpoint_ema_best.pt
```

Here are the results along with a pre-trained model on the ImageNet-K dataset. 

| Model | Parameters | Top-1 | Pre-trained weights | Config File |
| ---  | --- | --- | --- | --- |
| ResNet-50 | 25.5 M | 78.4 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/resnet50.pt) | [Link](../config/classification/resnet.yaml) |


Note that run-to-run variance of +/- 0.3 is natural on the ImageNet-1k dataset and could arise because of many factors, including different GPUs, seeds, etc.


