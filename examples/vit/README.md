# Vision Transformer (ViT)

[ViT](https://arxiv.org/abs/2010.11929) is a transformer-based models for visual recognition tasks. Note that our vision 
transformer model is different from the original ViT models in several aspects, including agnostic to input image scales. 
Please see [our RangeAugment paper](https://arxiv.org/abs/2212.10553) where we trained CLIP model with 
ViT-B/16 and ViT-H/16 with different input image resolutions.

We provide training and evaluation code along with pretrained models and configuration files for the following tasks:

1. [Image Classification on the ImageNet dataset](#imagenet-classification)
2. [Object detection on the MS-COCO dataset](#ms-coco-object-detection)
3. [Semantic segmentation on the ADE20k datasets](#ade20k-semantic-segmentation)


## ImageNet classification

### Training
Single node 8 A100 GPU training of [ViT-Base/16](./classification/vit_base.yaml) can be done using below command:

``` 
export CFG_FILE="examples/vit/classification/vit_base.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

### Evaluation and Results

We evaluate the model on a single GPU using following command:

```
 export CFG_FILE="examples/vit/classification/vit_base.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/classification/vit_base.pt"
 export DATASET_PATH="PATH_TO_DATASET"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CFG_FILE --common.results-loc classification_results --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

| Model    | Top-1 | Config                                   | Weights      |
|----------|-------|------------------------------------------|--------------|
| ViT-B/16 | 81.1  | [ViT-B/16](classification/vit_base.yaml) | [ViT-B/16](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/classification/vit_base.pt) |


## MS-COCO object detection

### Training ViT-B/16 with Mask R-CNN
Training ViTs for object detection task is expensive. We train our model on 8 nodes, each with 8 80 GB A100 GPUs. 
To train the detection model, we use pre-trained weights from CLIP ViT-B/16 model from [RangeAugment](https://arxiv.org/abs/2212.10553) paper. 
We train models with variable batch sampler in [MobileViT](https://arxiv.org/abs/2110.02178) paper and follow RangeAugment paper for data augmentation.

An example command for training on `i-th` node is
```
export CFG_FILE="examples/vit/detection/mask_rcnn_vit_base_clip.yaml"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE> # For Node-0, RANK=0; For Node-1, Rank=8, For Node-2, RANK=16, and so on.
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE> # WORLD_SIZE=8 nodes * 8 GPUS per node = 64
cvnets-train --common.config-file $CFG_FILE --common.results-loc results_detection --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

### Evaluation and results

We evaluate the model on a single GPU using following command:

```
 export CFG_FILE="examples/vit/detection/mask_rcnn_vit_base_clip.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/detection/mask_rcnn_vit_base_clip.pt"
 export DATASET_PATH="PATH_TO_DATASET"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-det --common.config-file $CFG_FILE \
 --common.results-loc seg_results \
 --model.detection.pretrained $MODEL_WEIGHTS --evaluation.detection.resize-input-images \
 --evaluation.detection.mode validation_set \
 --common.override-kwargs dataset.root_val=$DATASET_PATH
```

| Backbone | BBox mAP | Seg mAP | Config                                     | Weights      |
|----------|----------|---------|--------------------------------------------|--------------|
| ViT-B/16 | 50.7     | 44.2    | [ViT-B/16](detection/mask_rcnn_vit_base_clip.yaml)   | [ViT-B/16](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/detection/mask_rcnn_vit_base_clip.pt) |


## ADE20k semantic segmentation

## Training ViT-B/16 with DeepLabv3 on the ADE20k dataset

Training ViTs for semantic segmentation task is expensive. We train our model on a single node with 8 80 GB A100 GPUs. 
Similar to detection, we use pre-trained weights from CLIP ViT-B/16 model from [RangeAugment](https://arxiv.org/abs/2212.10553) paper. 
We follow RangeAugment paper for data augmentation.

We train model using below command:

``` 
export CFG_FILE="examples/vit/segmentation/ade20k/deeplabv3_vit_base_clip.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc segmentation_results
```
Note that we adjust the output stride of convolutional stem in ViT model so that the output from ViT backbone is 1/8th of the input size.


### Evaluation and results

Evaluation on the validation set can be done using the below command:

```
 export CFG_FILE="examples/vit/segmentation/ade20k/deeplabv3_vit_base_clip_os_8.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/segmentation/deeplabv3_vit_base_clip_os_8.pt"
 export DATASET_PATH="PATH_TO_DATASET"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file $CFG_FILE \
 --common.results-loc seg_results \
 --model.segmentation.pretrained $MODEL_WEIGHTS \
 --common.override-kwargs dataset.root_val=$DATASET_PATH
```

| Model  | Output stride | mIoU | Config                                                             | Weights                                                                                                    |
|--------|---------------|------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| ViT-B | 16            | 47.9 | [ViT-B/16](segmentation/ade20k/deeplabv3_vit_base_clip_os_16.yaml) | [ViT-B/16](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/segmentation/ade20k/deeplabv3_vit_base_clip_os_16.pt)      |
| ViT-B | 8             | 49.7 | [ViT-B/8](segmentation/ade20k/deeplabv3_vit_base_clip_os_8.yaml)   | [ViT-B/8](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/vit/segmentation/ade20k/deeplabv3_vit_base_clip_os_8.pt) |


## Citation

If you find our work useful, please cite following papers:

``` 
@inproceedings{dosovitskiy2021an,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=YicbFdNTTy}
}

@article{mehta2022rangeaugment,
  title={RangeAugment: Efficient Online Augmentation with Range Learning},
  author = {Mehta, Sachin and Naderiparizi, Saeid and Faghri, Fartash and Horton, Maxwell and Chen, Lailin and Farhadi, Ali and Tuzel, Oncel and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2212.10553},
  year={2022},
  url={https://arxiv.org/abs/2212.10553},
}
```
