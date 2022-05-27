# Training Classification Models on the ImageNet dataset

## Training on the ImageNet dataset

Single node training of any classification backbone can be done using below command:

``` 
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

By default, training will use all GPUs available on the machine. To restrict training to a subset of GPUs available on a machine, use `CUDA_VISIBLE_DEVICES` environment variable

For example configuration files, please see [config](../../../../../config/classification) folder. 

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

<details>
<summary>
Multi-node training of any classification backbone
</summary>

Assuming we have 4 8-GPU nodes (i.e., 32 GPUs), we can train  using below commands

Node-0
```
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --ddp.rank 0 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl
```
Node-1
```
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --ddp.rank 8 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl
```
Node-2
```
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --ddp.rank 16 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl
```

Node-3
```
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results --ddp.rank 24 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl
```
</details>

## Finetuning the classification model

If we want to finetune the ImageNet model with `1000` classes on another classification dataset with `N` classes, we can do so by using following arguments:
   * Pass this argument `--model.classification.finetune-pretrained-model` to enable finetuning
   * Specify number of classes in pre-trained model using `--model.classification.n-pretrained-classes` argument
   * Specify the location of pre-trained weights using `--model.classification.pretrained` argument

For a concrete example, see training recipe of [MobileViTv2](README-mobilevit-v2.md)


## Evaluating the classification model

Evaluation can be done using the below command:

```
 export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CFG_FILE --common.results-loc classification_results --model.classification.pretrained $MODEL_WEIGHTS
```

If you are evaluating the model from finetuning task, please disable `finetune-pretrained-model` argument by using the following `--common.override-kwargs model.classification.finetune_pretrained_model=false` argument
