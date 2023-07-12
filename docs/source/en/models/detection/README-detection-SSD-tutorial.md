# Object detection using SSDLite on MS-COCO

## Training detection network on the MS-COCO dataset

Single node training of SSD with any classification backbone, that adheres to [BaseEncoder](../../../../../cvnets/models/classification/base_cls.py) structure, can be done using the below command:

``` 
  export CONFIG_FILE="PATH_TO_CONFIG_FILE"
  export IMAGENET_PRETRAINED_WTS="LOCATION_OF_IMAGENET_WEIGHTS"
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file $CONFIG_FILE --common.results-loc ssd_results --model.classification.pretrained $IMAGENET_PRETRAINED_WTS
```

For example configuration files, please see [config](../../../../../config/detection/ssd_coco) folder. 

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Quantitative evaluation

Mean average precision score can be computed on MS-COCO dataset using the below command:

```
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export SSDLITE_MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS" 
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_results --model.detection.pretrained $SSDLITE_MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode validation_set
 ```

## Qualitative evaluation

An example command to run detection on an image using `SSDLite-MobileViTv2` model is given below
``` 
 export IMG_PATH="LOCATION_OF_IMAGE"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_mobilevitv2_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode single_image --evaluation.detection.path "${IMG_PATH}" --model.detection.ssd.conf-threshold 0.3
```

----

An example command to run detection on images stored in a folder using `SSDLite-MobileViTv2` model is given below
``` 
 export IMG_FOLDER_PATH="PATH_TO_FOLDER_CONTAINING_IMAGES"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_mobilevitv2_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode image_folder --evaluation.detection.path $IMG_FOLDER_PATH --model.detection.ssd.conf-threshold 0.3
```

## Example

In the below example, we download an image from the Internet and then detect objects using SSD w/ ResNet-50 backbone.  
``` 
 export IMG_PATH="http://farm3.staticflickr.com/2700/4532763002_7f4eafc5b0_z.jpg"
 export CFG_FILE="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.pt"
 cvnets-eval-det --common.config-file $CFG_FILE --common.results-loc ssdlite_results --model.detection.pretrained $MODEL_WEIGHTS --model.detection.n-classes 81 --evaluation.detection.resize-input-images --evaluation.detection.mode single_image --evaluation.detection.path "${IMG_PATH}" --model.detection.ssd.conf-threshold 0.3
```

## Citation

```
@inproceedings{liu2016ssd,
  title={Ssd: Single shot multibox detector},
  author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C},
  booktitle={European conference on computer vision},
  pages={21--37},
  year={2016},
  organization={Springer}
}
```
