# Training CLIP Models with RangeAugment

## Image-text dataset preparation

The dataset class for loading CLIP data is [here](../../data/datasets/multi_modal_img_text/img_text_tar_dataset.py). It requires data 
to be stored as multiple tar files in s3. In our set-up, each tar file contains about 1000 image-text pairs. 

An example of data organization is shown below, where `img_text_tar_dataset` folder stores two tar files, each file with 1000 pairs. 
Notice that we store image and text pairs in byte form. You may want to adapt the dataset class as per your needs.

```
img_text_tar_dataset/00000000_0_1000.tar.gz
|--- 00000000_0_image
|--- 00000000_0_text
|--- 00000000_1_image
|--- 00000000_1_text
|--- ...

img_text_tar_dataset/00000000_1000_2000.tar.gz
|--- 00000000_1000_image
|--- 00000000_1000_text
|--- 00000000_1001_image
|--- 00000000_1001_text
|--- ...
```

Once the dataset is in the format as described above, generate a `metadata file`. Metadata file is a dictionary 
storing the start image-text ids along with the tar file name. We used this metadata file for sharding data.

```
Example {'0-1000': '00000000_0_1000.tar.gz', 1000-2000', '00000000_1000_2000.tar.gz'}
```

## Training CLIP on image-text pair dataset

CLIP leverages our custom ViT implementation that can be used with multi-scale variable batch sampler. CLIP models are 
trained on multiple nodes, each node with multiple GPUs. Please see comments in configuration files for exact number of 
GPUs and nodes used in our experiments.


An example command for training on `i-th` node is
```
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE> # For Node-0, RANK=0; For Node-1, Rank=8, For Node-2, RANK=16, and so on.
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE> # WORLD_SIZE=32 nodes * 8 GPUS per node = 256
cvnets-train --common.config-file $CFG_FILE --common.results-loc results_clip --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Zero-shot evaluation of CLIP models on the ImageNet dataset

CLIP model can be evaluated on multiple input resolutions using below shell script:

```shell
#!/usr/bin/env bash

CONFIG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
IMAGENET_VAL_PATH="PATH_TO_IMAGENET_VAL_SET"
for res in 160 192 224 256 288; do  
    CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CONFIG_FILE --model.multi-modal-image-text.pretrained $MODEL_WEIGHTS \
    --model.text.transformer.classes-per-split-zero-shot 200 \
    --model.multi-modal-image-text.clip.cache-text-features-zero-shot \
    --common.override-kwargs image_augmentation.resize.size=$res image_augmentation.center_crop.size=$res \
    dataset.multi_modal_img_text.zero_shot.root_val=$IMAGENET_VAL_PATH
done
```

In the below table, we report the zero-shot top-1 accuracy of CLIP models trained with RangeAugment.

| Model            | 160   | 192   | 224   | 256   | 288   | Config                                          | Weights                                                                                             | Logs                                                                                                       |
|------------------|-------|-------|-------|-------|-------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| CLIP w/ ViT-B/16 | 69.26 | 71.07 | 71.84 | 72.34 | 72.82 | [CLIP-ViT-B/16_Config](clip/clip_vit_base.yaml) | [CLIP-ViT-B/16_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt) | [CLIP-ViT-B/16_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16_logs.txt) |
| CLIP w/ ViT-H/16 | 76.13 | 77.35 | 77.92 | 78.41 | 78.56 | [CLIP-ViT-H/16_Config](clip/clip_vit_huge.yaml) | [CLIP-ViT-H/16_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_huge_16.pt) | [CLIP-ViT-H/16_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_huge_16_logs.txt) |

***Note:*** For CLIP models, we found EMA and best checkpoints deliver similar performance. Here, we report the results for best checkpoint.


## Fine-tuning CLIP on the ImageNet dataset

Configuration files for fine-tuning clip model are [here](./clip_finetune_imagenet/). Please follow instructions for training and evaluation in the [classification readme file](README-classification.md).

We finetune the ViT backbone from CLIP model on the ImageNet dataset for 10 epochs. Below are the results:

| Model    | Top-1 @ 224x224 | Config                                                               | Weights         | Logs                                                                                                             |
|----------|-----------------|----------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------|
| ViT-B/16 | 84.31           | [CLIP-ViT-B/16_FT_Config](clip_finetune_imagenet/clip_vit_base.yaml) | [CLIP-ViT-B/16_FT_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_base_16_ft_in1k.pt) | [CLIP-ViT-B/16_FT_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_huge_16_ft_in1k_logs.txt) |
| ViT-H/16 | 86.90           | [CLIP-ViT-H/16_FT_Config](clip_finetune_imagenet/clip_vit_huge.yaml) | [CLIP-ViT-H/16_FT_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_huge_16_ft_in1k.pt) | [CLIP-ViT-H/16_FT_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_base_16_ft_in1k_logs.txt) |
