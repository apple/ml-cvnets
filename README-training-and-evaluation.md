# Training and Evaluating models using CVNets

***Note***
   * Sample configuration files that we used in our experiments are in [config](config) folder
   * Ensure that you edit location of training and validation set in the config files before evaluation.

## Training

   * Classification/Detection/Segmentation model can be trained as follows:
``` 
cvnets-train --common.config-file {config_file} --common.results-loc {results_loc}
```
where configuration file corresponding to the model is supplied using `--common.config-file` 

For example, to train a MobileVit-Small model on the ImageNet dataset, we can use the below command. Results will be stored in `results_mobilevit_small` folder
``` 
cvnets-train --common.config-file config/classification/mobilevit_small.yaml --common.results-loc results_mobilevit_small
```
Please see [config](config) folder for sample configurations for different tasks that are used in our experiments. 

## Evaluation

   * Classification models can be evaluating using the following command

```
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc "${EVAL_DIR}" --model.classification.pretrained "${EVAL_DIR}/${CKPT_NAME}"
```

   * Detection models can be evaluated on the validation set using the following command. Because detection models are trained for a specific input image resolution, we use `--evaluation.detection.resize-input-images` as an argument to resize the input images to the same size that is used during training. 

``` 
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc "${EVAL_DIR}" --model.detection.pretrained "${EVAL_DIR}/${CKPT_NAME}" \
    --evaluation.detection.mode validation_set --model.detection.n-classes 81 --evaluation.detection.resize-input-images
```


   * Similar to the detection model, segmentation models can be evaluated using the following command:

``` 
export EVAL_DIR='' # Location of results 
export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
export N_CLASSES=21 # number of classes for the pascal voc dataset
CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file "${EVAL_DIR}/config.yaml" --common.results-loc "${EVAL_DIR}" --model.segmentation.pretrained "${EVAL_DIR}/${CKPT_NAME}" \
    --evaluation.segmentation.mode validation_set --model.segmentation.n-classes $N_CLASSES --evaluation.segmentation.resize-input-images
```