# Config Files: Introduction and Walkthrough

Config files in CVNet are stored as YAML files and stored under `config/<task>` directories.
They contain the hyper-parameters used for training/validating the respective models.

Let us take a step-by-step look at `config/classification/imagenet/resnet.yaml` which is used to train a ResNet-50 model on the ImageNet-1k dataset using a single node with 8 A100 GPUs:

## Dataset

The configs under `dataset` define which dataset to train (`dataset.name`) and where the data is located on disk (`dataset.root_train` and `dataset.root_val`).
It also contains information about the train/val batch sizes and the number of workers and how to handle GPU memory. Note that the effective batch size is `train_batch_size0 * num_gpus * gradient accum. freq.`

```yaml
dataset:
  root_train: "/mnt/imagenet/training"
  root_val: "/mnt/imagenet/validation"
  name: "imagenet"
  category: "classification"
  train_batch_size0: 128
  val_batch_size0: 100
  eval_batch_size0: 100
  workers: 8
  persistent_workers: true
  pin_memory: true
```

## Data augmentation

The `image_augmentation` configs define the data augmentations to use during training. In below example, we use [Inception-style](https://arxiv.org/abs/1409.4842) augmentation. 
For advanced image augmentation example, see [ResNet-50's advanced recipe](../../../../config/classification/imagenet/resnet_adv.yaml).

```yaml
image_augmentation:
  random_resized_crop:
    enable: true
    interpolation: "bicubic"
  random_horizontal_flip:
    enable: true
  resize:
    enable: true
    size: 256 # shorter size is 256
    interpolation: "bicubic"
  center_crop:
    enable: true
    size: 224 
```

## Sampler

The `sampler` configs define which Data Sampler type to use as well as information about the crop width and height. In this example, we train ResNet-50 with variably-sized batch sampler, introduced in [MobileViT](https://arxiv.org/abs/2110.02178).

```yaml
sampler:
  name: "variable_batch_sampler"
  vbs:
    crop_size_width: 224
    crop_size_height: 224
    max_n_scales: 5
    min_crop_size_width: 128
    max_crop_size_width: 320
    min_crop_size_height: 128
    max_crop_size_height: 320
    check_scale: 32
```

## Optimizer and LR scheduler

The `optim` and `scheduler` configs define the optimizer and LR scheduler hyper-parameters. Here we used SGD with a Consine learning rate with warm-up.

```yaml
optim:
  name: "sgd"
  weight_decay: 1.e-4
  no_decay_bn_filter_bias: true
  sgd:
    momentum: 0.9
scheduler:
  name: "cosine"
  is_iteration_based: false
  max_epochs: 150
  warmup_iterations: 7500
  warmup_init_lr: 0.05
  cosine:
    max_lr: 0.4
    min_lr: 2.e-4
```

## Model

`model` defines the model type as well as the model hyper-parameters. Here Used a `ResNet-50` model for a classification task.

```yaml
model:
  classification:
    name: "resnet"
    activation:
      name: "relu"
    resnet:
      depth: 50
  normalization:
    name: "batch_norm"
    momentum: 0.1
  activation:
    name: "relu"
    inplace: true
  layer:
    global_pool: "mean"
    conv_init: "kaiming_normal"
    linear_init: "normal"
```

## EMA and Training statistics

CVNet allows you to keep an exponentially moving average version of the model by simply setting `ema.enable = True`. Last but not least, `stats` defines which metrics to compute and report for the model. The best model is kept based on its `checkpoint_metric` value.

```yaml
ema:
  enable: true
  momentum: 0.0005
stats:
  val: [ "loss", "top1", "top5" ]
  train: ["loss"]
  checkpoint_metric: "top1"
  checkpoint_metric_max: true
```

