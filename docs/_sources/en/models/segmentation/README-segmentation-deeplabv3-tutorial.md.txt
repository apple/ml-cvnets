# Semantic Segmentation using DeepLabv3

## Training segmentation network

Single node training of DeepLabv3 with any classification backbone, that adheres to [BaseEncoder](../../../../../cvnets/models/classification/base_cls.py) structure, can be done using the below command:

``` 
  export CONFIG_FILE="PATH_TO_CONFIG_FILE"
  export IMAGENET_PRETRAINED_WTS="LOCATION_OF_IMAGENET_WEIGHTS"
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file $CONFIG_FILE --common.results-loc deeplabv3_results --model.classification.pretrained $IMAGENET_PRETRAINED_WTS
```

For example configuration files, please see [config](../../../../../config/segmentation) folder. 

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Quantitative evaluation

Mean intersection over union (mIoU) score can be computed on segmentation dataset using the below command:

```
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export DEEPLABV3_MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS" 
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $DEEPLABV3_MODEL_WEIGHTS --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode validation_set
 ```

## Qualitative evaluation

An example command to run segmentation on an image using `DeepLabv3-MobileNetv2` model trained on the PASCAL VOC 2012 is given below
``` 
 export IMG_PATH="LOCATION_OF_IMAGE"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $MODEL_WEIGHTS --model.segmentation.n-classes 21 --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode single_image --evaluation.segmentation.path "${IMG_PATH}" \
 --evaluation.segmentation.save-masks \
 --evaluation.segmentation.apply-color-map \
 --evaluation.segmentation.save-overlay-rgb-pred
```

Notes:
   * `--evaluation.segmentation.save-masks` option saves segmentation masks whose values range between 0 and `NUM_CLASSES - 1`
   * `--evaluation.segmentation.apply-color-map` option applies color map to segmentation masks and saves them
   * `--evaluation.segmentation.save-overlay-rgb-pred` option overlays the predicted segmentation mask over rgb images. To adjust the overlay alpha, use `--evaluation.segmentation.overlay-mask-weight` option, which defaults to `0.5`.

----

An example command to run segmentation on images stored in a folder using `DeepLabv3-MobileNetv2` model trained on the PASCAL VOC 2012 is given below
``` 
 export IMG_FOLDER_PATH="PATH_TO_FOLDER_CONTAINING_IMAGES"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $MODEL_WEIGHTS --model.segmentation.n-classes 21 --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode image_folder --evaluation.segmentation.path $IMG_FOLDER_PATH \
 --evaluation.segmentation.save-masks \
 --evaluation.segmentation.apply-color-map \
 --evaluation.segmentation.save-overlay-rgb-pred
```

## Example

In the below example, we download an image from the Internet and then segment objects using DeepLabv3 w/ [MobileViT](https://arxiv.org/abs/2110.02178) backbone.  
``` 
 export IMG_PATH="http://farm7.staticflickr.com/6206/6118204766_b1c9a39153_z.jpg"
 export CFG_FILE="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/deeplabv3-mobilevitv1.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/deeplabv3-mobilevitv1.pt"
 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $MODEL_WEIGHTS --model.segmentation.n-classes 21 \
 --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode single_image --evaluation.segmentation.path "${IMG_PATH}" --evaluation.segmentation.save-masks \
 --evaluation.segmentation.apply-color-map --evaluation.segmentation.save-overlay-rgb-pred
```

## Citation

```
@article{chen2017rethinking,
  title={Rethinking atrous convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={arXiv preprint arXiv:1706.05587},
  year={2017}
}
```
