# Semantic segmentation usng MobileViTv2

## Training segmentation network

Single node 4-GPU training of `DeepLabv3-MobileViTv2-1.0` on the PASCAL VOC 2012 dataset.

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/pascal_voc/deeplabv3_mobilevitv2.yaml --common.results-loc deeplabv3_mobilevitv2_results/width_1_0_0 --common.override-kwargs model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
where `LOCATION_OF_IMAGENET_1k_CHECKPOINT` is the location of the best EMA checkpoint of the MobileViTv2 model pretrained on the ImageNet-1k dataset.

<details>
<summary>
Single node 4-GPU training of PSPNet-MobileViTv2-1.0 on the PASCAL VOC 2012 dataset.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/pascal_voc/pspnet_mobilevitv2.yaml --common.results-loc pspnet_mobilevitv2_results/width_1_0_0 --common.override-kwargs model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node 4-GPU training of DeepLabv3-MobileViTv2-1.0 on the ADE20k dataset.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/ade20k/deeplabv3_mobilevitv2.yaml --common.results-loc deeplabv3_ade20k_results/width_1_0_0 --common.override-kwargs model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node 4-GPU training of DeepLabv3-MobileViTv2-1.0 on the ADE20k dataset.
</summary>

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/ade20k/pspnet_mobilevitv2.yaml --common.results-loc pspnet_ade20k_results/width_1_0_0 --common.override-kwargs model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>



## Quantitative evaluation

Mean intersection over union (mIoU) score can be computed on segmentation dataset using the below command:

```
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export DEEPLABV3_MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS" 
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc seg_results --model.segmentation.pretrained $DEEPLABV3_MODEL_WEIGHTS --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode validation_set
 ```

## Qualitative evaluation
In the below example, we download an image from the Internet and then segment objects using DeepLabv3 w/ MobileViTv2-1.0 backbone.

``` 
 export IMG_PATH="http://farm7.staticflickr.com/6206/6118204766_b1c9a39153_z.jpg"
 export CFG_FILE="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.0.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt"
 cvnets-eval-seg 
 python main_eval.py --common.config-file $CFG_FILE --common.results-loc deeplabv3_mobilevitv2_results --model.segmentation.pretrained $MODEL_WEIGHTS --model.segmentation.n-classes 21 \
 --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode single_image --evaluation.segmentation.path "${IMG_PATH}" --evaluation.segmentation.save-masks \
 --evaluation.segmentation.apply-color-map --evaluation.segmentation.save-overlay-rgb-pred
```

## Citation

``` 
@article{mehta2022separable,
  title={Separable Self-attention for Mobile Vision Transformers},
  author={Sachin Mehta and Mohammad Rastegari},
  year={2022}
}
```
