# Object detection using SSDLite on MS-COCO

## Training detection network on the MS-COCO dataset

Single node training of `SSDLite-MobileViTv2-2.0` with 4 A100 GPUs.

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_2_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
where `LOCATION_OF_IMAGENET_1k_CHECKPOINT` is the location of the best EMA checkpoint of the MobileViTv2 model pretrained on the ImageNet-1k dataset.

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-1.75` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_1_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-1.5` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_1_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-1.25` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_1_2_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.25 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-1.0` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_1_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-0.75` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_0_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=0.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node training of `SSDLite-MobileViTv2-0.5` with 4 A100 GPUs.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/ssd_coco/mobilevit_v2.yaml --common.results-loc ssdlite_mobilevitv2_results/width_0_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=0.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

## Quantitative evaluation

Mean average precision score can be computed on MS-COCO dataset using the below command:

```
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_mobilevitv2_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode validation_set
 ```

## Qualitative evaluation

An example command to run detection on an image using `SSDLite-MobileViTv2` model is given below
``` 
 export IMG_PATH="LOCATION_OF_IMAGE_FILE"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_mobilevitv2_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode single_image --evaluation.detection.path "${IMG_PATH}" --model.detection.ssd.conf-threshold 0.3
```

An example command to run detection on images stored in a folder using `SSDLite-MobileViTv2` model is given below
``` 
 export IMG_FOLDER_PATH="PATH_TO_FOLDER_CONTAINING_IMAGES"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_mobilevitv2_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode image_folder --evaluation.detection.path $IMG_FOLDER_PATH --model.detection.ssd.conf-threshold 0.3
```

## Citation

``` 
@article{mehta2022separable,
  title={Separable Self-attention for Mobile Vision Transformers},
  author={Sachin Mehta and Mohammad Rastegari},
  year={2022}
}
```
