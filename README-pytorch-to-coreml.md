# Converting models trained using CVNets to CoreML

For conversion, we assume that you are using `MAC OS` machine.

## Classification networks

We can convert the classification models using the following command

```
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
cvnets-convert --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc $EVAL_DIR --model.classification.pretrained "${EVAL_DIR}/${CKPT_NAME}"  --conversion.coreml-extn mlmodel
```

## Detection networks

We can convert the detection models trained on MS-COCO (81 classes, including background) using the following command

```
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
cvnets-convert --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc $EVAL_DIR --model.detection.pretrained "${EVAL_DIR}/${CKPT_NAME}"  --conversion.coreml-extn mlmodel --model.detection.n-classes 81
```

## Segmentation networks

We can convert the segmentation models trained on the PASCAL VOC 2012 dataset (21 classes, including background) using the following command

```
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
cvnets-convert --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc $EVAL_DIR --model.segmentation.pretrained "${EVAL_DIR}/${CKPT_NAME}"  --conversion.coreml-extn mlmodel --model.segmentation.n-classes 21
```