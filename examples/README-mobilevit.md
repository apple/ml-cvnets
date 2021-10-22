# Training MobileViT Models

MobileViT models can be trained for three tasks:
   * ImageNet-1k Classification
   * Object detection
   * Semantic Segmentation

Configuration files of MobileViT models used for experiments in the paper for each task are located inside [config](../config/classification) folder. You may need to edit these files depending upon the location of datasets, compute infrastructure, and other factors.

## ImageNet-1k classification

We trained MobileViT models on the ImageNet-1k dataset using 8 NVIDIA A100 GPUs. On an average, training takes about 2 days.

### Training

Training MobileViT models is simple. For example, MobileViT-S model can be trained by running the following command from the root directory:
``` 
cvnets-train --common.config-file config/classification/mobilevit_small.yaml --common.results-loc results_imagenet1k_mobilevit_s
```

Note that the default location of the ImageNet-1k training and validation sets in the configuration file are `/mnt/imagenet/training` and `/mnt/imagenet/validation` respectively. Please make changes accordingly.

Configuration files for different MobileViT models are:
   * [MobileViT-S](../config/classification/mobilevit_small.yaml)
   * [MobileViT-XS](../config/classification/mobilevit_x_small.yaml)
   * [MobileViT-XXS](../config/classification/mobilevit_xx_small.yaml)

### Evaluation
Evaluation on the ImageNet-1k dataset can be achieved using `cvnets-eval` command. Let us assume that we want to evaluate MobileViT-S model whose weight file name is `checkpoint_ema.pt` and is stored in `results_imagenet1k_mobilevit_s` folder. Also, this folder should also contain `config.yaml` file, which is nothing but a copy of the configuration file that was used during training. 

Now, we can run the following command to evaluate the performance on GPU-0:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file results_imagenet1k_mobilevit_s/config.yaml --common.results-loc results_imagenet1k_mobilevit_s --model.classification.pretrained results_imagenet1k_mobilevit_s/checkpoint_ema_best.pt
```

### Results
Below are the results on the ImageNet-K dataset.

| Model | Parameters | Top-1 | Pre-trained weights | Config File |
| ---  | --- | --- | --- | --- |
| MobileViT-XXS | 1.3 M | 69.0 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.yaml) |
| MobileViT-XS | 2.3 M | 74.7 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.yaml) |
| MobileViT-S | 5.6 M | 78.3 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.yaml) |

Note that run-to-run variance of +/- 0.3 is natural on the ImageNet-1k dataset and could arise because of many factors, including different GPUs, seeds, etc.

## Object detection

We trained MobileViT models on the ImageNet-1k dataset using 4 NVIDIA A100 GPUs. On an average, training takes about 2 days.

### Training

Similar to classification models, training SSDLite models with MobileViT as a backbone is simple. For example, SSDLite with MobileViT-S model can be trained by running the following command from the root directory:
``` 
cvnets-train --common.config-file config/detection/ssd_mobilevit_small_320.yaml --common.results-loc results_coco_mobilevit_s
```

Note that the default location of the MS-COCO training and validation sets in the configuration file are set to `/mnt/vision_datasets/coco`. Please make changes in the configuration files accordingly.

Configuration files for different SSDLite models with MobileViT as a backbone are:
   * [MobileViT-S](../config/detection/ssd_mobilevit_small_320.yaml)
   * [MobileViT-XS](../config/detection/ssd_mobilevit_x_small_320.yaml)
   * [MobileViT-XXS](../config/detection/ssd_mobilevit_xx_small_320.yaml)

### Evaluation
Below are three options supported for evaluation:

   * **MS-COCO Validation set:** Evaluation on the MS-COCO validation set can be achieved using `cvnets-eval-det` command. Let us assume that we want to evaluate MobileViT-S model whose weight file name is `checkpoint_ema.pt` and is stored in `results_coco_mobilevit_s` folder. Also, this folder should also contain `config.yaml` file, which is nothing but a copy of the configuration file that was used during training. 

Now, we can run the following command to evaluate the performance on GPU-0:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file results_coco_mobilevit_s/config.yaml --common.results-loc results_coco_mobilevit_s --model.detection.pretrained results_coco_mobilevit_s/checkpoint_ema_best.pt --evaluation.detection.mode validation_set --evaluation.detection.resize-input-images
```

   * **Detection on images stored in a folder:** To evaluate on a set of images stored in a directory say `sample_images`, you can run the following command:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file results_coco_mobilevit_s/config.yaml --common.results-loc results_coco_mobilevit_s --model.detection.pretrained results_coco_mobilevit_s/checkpoint_ema_best.pt \
--evaluation.detection.mode image_folder --model.detection.n-classes 81 --evaluation.detection.path sample_images --model.detection.ssd.conf-threshold 0.3
```

   * **Detection on a single image:** To evaluate on a single image called `sample_image.jpg`, you can run the following command:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file results_coco_mobilevit_s/config.yaml --common.results-loc results_coco_mobilevit_s --model.detection.pretrained results_coco_mobilevit_s/checkpoint_ema_best.pt \
--evaluation.detection.mode single_image --model.detection.n-classes 81 --evaluation.detection.path sample_image.jpg --model.detection.ssd.conf-threshold 0.3
```

***Note***: In the default configuration file that we used during training, we have specified the location of pre-trained weights of the backbone model using `model.classification.pretrained` key. Because this file is not required during evaluation, please set it to an empty string.


### Results
Below are the results on the MS-COCO validation set.

| Backbone | Parameters | mAP | Pre-trained weights | Config file |
| ---  | --- | --- | --- |  --- |
| MobileViT-XXS | 1.8 M | 19.9 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_xxs_320.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_xxs_320.yaml) |
| MobileViT-XS | 2.7 M | 24.8 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_xs_320.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_xs_320.yaml) |
| MobileViT-S | 5.7 M | 27.9 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_s_320.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/detection/ssd_mobilevit_s_320.yaml) |

## Semantic segmentation

We trained MobileViT models on the PASCAL VOC dataset using 4 NVIDIA A100 GPUs. On an average, training takes about half a day.

### Training

Similar to classification models, training DeepLabv3 models with MobileViT as a backbone is simple. For example, DeepLabv3 with MobileViT-S model can be trained by running the following command from the root directory:
``` 
cvnets-train --common.config-file config/segmentation/deeplabv3_mobilevit_small.yaml --common.results-loc results_voc_mobilevit_s
```

Note that the default location of the MS-COCO training and validation sets in the configuration file are set to `/mnt/vision_datasets/pascal_voc/VOCdevkit/`. Please make changes in the configuration files accordingly.

Configuration files for different DeepLabv3 models with MobileViT as a backbone are:
   * [MobileViT-S](../config/segmentation/deeplabv3_mobilevit_small.yaml)
   * [MobileViT-XS](../config/segmentation/deeplabv3_mobilevit_x_small.yaml)
   * [MobileViT-XXS](../config/segmentation/deeplabv3_mobilevit_xx_small.yaml)

### Evaluation

Below are three options supported for evaluation:

   * **Validation set:** Evaluation on the validation set can be achieved using `cvnets-eval-seg` command. Let us assume that we want to evaluate MobileViT-S model whose weight file name is `checkpoint_ema.pt` and is stored in `results_voc_mobilevit_s` folder. Also, this folder should also contain `config.yaml` file, which is nothing but a copy of the configuration file that was used during training. 

Now, we can run the following command to evaluate the performance on GPU-0:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file results_voc_mobilevit_s/config.yaml --common.results-loc results_voc_mobilevit_s --model.segmentation.pretrained results_voc_mobilevit_s/checkpoint_ema_best.pt --evaluation.segmentation.mode validation_set --evaluation.segmentation.resize-input-images
```

   * **Segmentation on images stored in a folder:** To evaluate on a set of images stored in a directory say `sample_images`, you can run the following command:
``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file results_voc_mobilevit_s/config.yaml --common.results-loc results_voc_mobilevit_s --model.segmentation.pretrained results_voc_mobilevit_s/checkpoint_ema_best.pt \
--evaluation.segmentation.mode image_folder --model.segmentation.n-classes 21 --evaluation.segmentation.path sample_images --evaluation.segmentation.save-masks 
```
***Note 1:*** Color segmentation masks can be obtained by adding `--evaluation.segmentation.apply-color-map` argument to the above command. 

***Note 2:*** Segmentation masks overlayed on original images can be obtained by adding `--evaluation.segmentation.save-overlay-rgb-pred` argument to the above command.

   * **Segmenting a single image:** To segment a single image called `sample_image.jpg`, you can run the following command:

``` 
CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file results_voc_mobilevit_s/config.yaml --common.results-loc results_voc_mobilevit_s --model.segmentation.pretrained results_voc_mobilevit_s/checkpoint_ema_best.pt \
--evaluation.segmentation.mode single_image --model.segmentation.n-classes 21 --evaluation.segmentation.path sample_image.jpg --evaluation.segmentation.save-masks --evaluation.segmentation.apply-color-map --evaluation.segmentation.save-overlay-rgb-pred
```

***Note***: In the default configuration file that we used during training, we have specified the location of pre-trained weights of the backbone model using `model.classification.pretrained` key. Because this file is not required during evaluation, please set it to an empty string.

### Results
Below are the results on the PASCAL VOC 2012 validation set.

| Backbone | Parameters | mAP | Pre-trained weights | Config file |
| ---  | --- | --- | --- | --- |
| MobileViT-XXS | 1.9 M | 72.8 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_xxs.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_xxs.yaml) |
| MobileViT-XS | 2.9 M | 77.1 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_xs.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_xs.yaml) |
| MobileViT-S | 6.4 M | 79.3 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_s.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/segmentation/deeplabv3_mobilevit_s.yaml) |