# Converting models trained using CVNets to CoreML

For conversion, we assume that you are using `MAC OS` machine. We follow [CoreML](https://coremltools.readme.io/docs/pytorch-conversion) tutorial, i.e., 
first convert the PyTorch model to JIT, and then convert to CoreML model.

## Classification networks

We can convert the classification models using the following command

```
export CONFIG_FILE="LOCATION_OF_CONFIG_FILE"
export MODEL_WEIGHTS="LOCATION_OF_MODEL_WEIGHT_FILE"
cvnets-convert --common.config-file $CONFIG_FILE --common.results-loc coreml_models_cls --model.classification.pretrained $MODEL_WEIGHTS  --conversion.coreml-extn mlmodel
```

## Detection networks

We can convert the detection models trained on MS-COCO (81 classes, including background) using the following command

```
export CONFIG_FILE="LOCATION_OF_CONFIG_FILE"
export MODEL_WEIGHTS="LOCATION_OF_MODEL_WEIGHT_FILE"
export N_CLASSES="NUMBER_OF_CLASSES"
cvnets-convert --common.config-file $CONFIG_FILE --common.results-loc coreml_models_det --model.detection.pretrained $MODEL_WEIGHTS --conversion.coreml-extn mlmodel --model.detection.n-classes $N_CLASSES
```

## Segmentation networks

We can convert the segmentation models using the following command

```
export CONFIG_FILE="LOCATION_OF_CONFIG_FILE"
export MODEL_WEIGHTS="LOCATION_OF_MODEL_WEIGHT_FILE"
export N_CLASSES="NUMBER_OF_CLASSES"
cvnets-convert --common.config-file $CONFIG_FILE --common.results-loc coreml_models_res --model.segmentation.pretrained $MODEL_WEIGHTS --conversion.coreml-extn mlmodel --model.segmentation.n-classes $N_CLASSES
```

## Example to convert MobileViTv2 model

We can convert `MobileViTv2-1.0` classification model trained on ImageNet-1k dataset using below commands:
``` 
export CONFIG_FILE="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.yaml"
export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.pt"
cvnets-convert --common.config-file $CONFIG_FILE --common.results-loc coreml_models_cls --model.classification.pretrained $MODEL_WEIGHTS  --conversion.coreml-extn mlmodel
```
