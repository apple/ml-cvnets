# Training MobileViTv2 Models

## ImageNet-1k Training
Single node training of `MobileViTv2-2.0` with 8 A100 GPUs. On an average, training takes about 2 days.

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_2_0_0 --common.override-kwargs scheduler.cosine.max_lr=0.0020 scheduler.cosine.min_lr=0.0002  optim.weight_decay=0.050 model.classification.mitv2.width_multiplier=2.00
```

   * You may need to change the `training` and `validation` dataset locations in the config file
   * We train each model using `8 A100` GPUs with an effective batch size of 1024 images (128 images per GPU x 8 GPUs)
   * If you are not using A100 GPUs and getting OOM errors, you may try one of the following options:
      * Reduce the batch size and increase number of GPUs, so that effective batch size is 1024 images.
      * Reduce the batch size and use gradient accumulation option (`--common.accum-freq`), so that effective batch size (`= images per GPU` * `number of GPUs` * `--common.accum-freq`) is 1024 images. 

----

Evaluation can be done using the below command:

``` 
    CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_2_0_0 --model.classification.pretrained mobilevitv2_results/width_2_0_0/checkpoint_ema_best.pt --common.override-kwargs model.classification.mitv2.width_multiplier=2.00
```

Examples are given below to train other variants of MobileViTv2 model. ***Note*** that we linearly (1) increase the learning rate and (2) decay the weight decay with respect to MobileViTv2-2.0 configuration.  

<details>
<summary>
MobileViTv2-1.75
</summary>
Single node training of MobileViTv2-1.75 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_1_7_5 --common.override-kwargs scheduler.cosine.max_lr=0.0026 scheduler.cosine.min_lr=0.00026 optim.weight_decay=0.039 model.classification.mitv2.width_multiplier=1.75
```
</details>

<details>
<summary>
MobileViTv2-1.5
</summary>
Single node training of MobileViTv2-1.5 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_1_5_0 --common.override-kwargs scheduler.cosine.max_lr=0.0035 scheduler.cosine.min_lr=0.00035 optim.weight_decay=0.029 model.classification.mitv2.width_multiplier=1.50
```
</details>

<details>
<summary>
MobileViTv2-1.25
</summary>
Single node training of MobileViTv2-1.25 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_1_2_5 --common.override-kwargs scheduler.cosine.max_lr=0.0049 scheduler.cosine.min_lr=0.00049 optim.weight_decay=0.020 model.classification.mitv2.width_multiplier=1.25
```
</details>

<details>
<summary>
MobileViTv2-1.00
</summary>
Single node training of MobileViTv2-1.00 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_1_0_0 --common.override-kwargs scheduler.cosine.max_lr=0.0075 scheduler.cosine.min_lr=0.00075 optim.weight_decay=0.013 model.classification.mitv2.width_multiplier=1.00
```
</details>

<details>
<summary>
MobileViTv2-0.75
</summary>
Single node training of MobileViTv2-0.75 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_0_7_5 --common.override-kwargs scheduler.cosine.max_lr=0.0090 scheduler.cosine.min_lr=0.00090 optim.weight_decay=0.008 model.classification.mitv2.width_multiplier=0.75
```
</details>

<details>
<summary>
MobileViTv2-0.5
</summary>
Single node training of MobileViTv2-0.5 with 8 A100 GPUs

``` 
 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet/mobilevit_v2.yaml --common.results-loc mobilevitv2_results/width_0_5_0 --common.override-kwargs scheduler.cosine.max_lr=0.0090 scheduler.cosine.min_lr=0.00090 optim.weight_decay=0.004 model.classification.mitv2.width_multiplier=0.50
```
</details>

## ImageNet-21k-P Pre-training

ImageNet-21k-P is significantly larger than ImageNet-1k in terms of dataset size. To train on ImageNet-21k-P, we use 32 A100 GPUs, i.e., 4 8-GPU nodes. For faster convergence, we initialize the model with ImageNet-1k checkpoint.

Multi-node training of MobileViTv2-2.0 with 4 8-GPU nodes on the ImageNet-21k-P dataset

Node-0
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_2_0_0 --ddp.rank 0 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.05 model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-1
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_2_0_0 --ddp.rank 8 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.05 model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-2
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_2_0_0 --ddp.rank 16 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.05 model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```

Node-3
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_2_0_0 --ddp.rank 24 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.05 model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```

---- 

Evaluation can be done using the below command:

``` 
  CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_2_0_0 --model.classification.pretrained mobilevitv2_results_in21k/checkpoint_ema_best.pt --common.override-kwargs model.classification.mitv2.width_multiplier=2.00 model.classification.finetune_pretrained_model=false
```

<details>
<summary>
Multi-node training of MobileViTv2-1.75 with 4 8-GPU nodes on the ImageNet-21k-P dataset
</summary>

Node-0
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_7_5 --ddp.rank 0 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.039 model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-1
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_7_5 --ddp.rank 8 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.039 model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-2
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_7_5 --ddp.rank 16 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.039 model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```

Node-3
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_7_5 --ddp.rank 24 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.039 model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Multi-node training of MobileViTv2-1.5 with 4 8-GPU nodes on the ImageNet-21k-P dataset
</summary>

Node-0
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_5_0 --ddp.rank 0 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.029 model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-1
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_5_0 --ddp.rank 8 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.029 model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
Node-2
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_5_0 --ddp.rank 16 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.029 model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```

Node-3
```
cvnets-train --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k/width_1_5_0 --ddp.rank 24 --ddp.world-size 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.backend nccl --common.override-kwargs optim.weight_decay=0.029 model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

## ImageNet-1k Fine-tuning

### Finetune from ImageNet-1k pre-training at 384x384 resolution

Single node finetuning of `MobileViTv2-2.0` with 2 A100 GPUs
```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_2_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
where `LOCATION_OF_IMAGENET_1k_CHECKPOINT` is the location of the best EMA checkpoint obtained after pre-training on the ImageNet-1k dataset at 256x256 resolution.

----

Evaluation can be done using the below command:

``` 
  CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file config/classification/imagenet_21k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_2_0_0 --model.classification.pretrained mobilevitv2_results_in1k_ft_384/width_2_0_0/checkpoint_ema_best.pt --common.override-kwargs model.classification.mitv2.width_multiplier=2.00
```

<details>
<summary>
Single node finetuning of MobileViTv2-1.75 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_1_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-1.5 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_1_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-1.25 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_1_2_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.25 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-1.00 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_1_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.00 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-0.75 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_0_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=0.75 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-0.5 with 2 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in1k.yaml --common.results-loc mobilevitv2_results_in1k_ft_384/width_0_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=0.5 model.classification.pretrained="LOCATION_OF_IMAGENET_1k_CHECKPOINT"
```
</details>

***Note***: If node has more than 2 GPUs, then `CUDA_VISIBLE_DEVICES=0,1` can be used to run job on GPU-0 and GPU-1 only.  

### Finetune from ImageNet-21k-P pre-training at 256x256 resolution

Single node finetuning of `MobileViTv2-2.0` with 4 A100 GPUs
```
cvnets-train --common.config-file config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k_ft_256/width_2_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_CHECKPOINT"
```
where `LOCATION_OF_IMAGENET_21k_CHECKPOINT` is the location of the best EMA checkpoint obtained after pre-training on the ImageNet-21k-P dataset at 256x256 resolution.

----

Evaluation can be done using the below command:

``` 
  CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k_ft_256/width_2_0_0 --model.classification.pretrained mobilevitv2_results_in21k_ft_256/width_2_0_0/checkpoint_ema_best.pt --common.override-kwargs model.classification.mitv2.width_multiplier=2.00 model.classification.finetune_pretrained_model=false
```

<details>
<summary>
Single node finetuning of MobileViTv2-1.75 with 4 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k_ft_256/width_1_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-1.5 with 4 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml --common.results-loc mobilevitv2_results_in21k_ft_256/width_1_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_CHECKPOINT"
```
</details>


#### Finetune from ImageNet-21k-P+ImageNet-1k pre-training at 384x384 resolution

Single node finetuning of `MobileViTv2-2.0` with 2 A100 GPUs
```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in21k_in1k.yaml --common.results-loc mobilevitv2_results_in21k_1k_ft_384/width_2_0_0 --common.override-kwargs model.classification.mitv2.width_multiplier=2.0 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_1k_CHECKPOINT"
```

where `LOCATION_OF_IMAGENET_21k_1k_CHECKPOINT` is the location of the best EMA checkpoint obtained after finetuning a pre-trained ImageNet-21k-P model on ImageNet-1k at 256x256 resolution.

----

Evaluation can be done using the below command:

```
  CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in21k_in1k.yaml --common.results-loc mobilevitv2_results_in21k_1k_ft_384/width_2_0_0 --model.classification.pretrained mobilevitv2_results_in21k_1k_ft_384/width_2_0_0/checkpoint_ema_best.pt --common.override-kwargs model.classification.mitv2.width_multiplier=2.00 model.classification.finetune_pretrained_model=false
```

<details>
<summary>
Single node finetuning of MobileViTv2-1.75 with 4 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in21k_in1k.yaml --common.results-loc mobilevitv2_results_in21k_1k_ft_384/width_1_7_5 --common.override-kwargs model.classification.mitv2.width_multiplier=1.75 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_1k_CHECKPOINT"
```
</details>

<details>
<summary>
Single node finetuning of MobileViTv2-1.5 with 4 A100 GPUs
</summary>

```
cvnets-train --common.config-file config/classification/finetune_higher_res/mobilevit_v2_in21k_in1k.yaml --common.results-loc mobilevitv2_results_in21k_1k_ft_384/width_1_5_0 --common.override-kwargs model.classification.mitv2.width_multiplier=1.5 model.classification.pretrained="LOCATION_OF_IMAGENET_21k_1k_CHECKPOINT"
```
</details>


## Citation

``` 
@article{mehta2022separable,
  title={Separable Self-attention for Mobile Vision Transformers},
  author={Sachin Mehta and Mohammad Rastegari},
  year={2022}
}
```
