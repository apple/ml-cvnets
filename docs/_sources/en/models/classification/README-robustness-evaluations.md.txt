# Evaluating classification models on ImageNet distribution shift datasets

We support evaluating on ImageNet-A/R/Sketch datasets. For evaluation, download 
these datasets and set `root_val` path to the root directory of the dataset.

## Downloading datasets
Please follow instructions for each datasets from the following links:
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)


## Evaluating a classification model

Evaluation can be done using the below command.  Please set the `root_val` in 
the configuration to the dataset path.

```
 export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CFG_FILE --common.results-loc classification_results --model.classification.pretrained $MODEL_WEIGHTS
```
