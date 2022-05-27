# Training MobileViT for action recognition

## Training MobileViT on Kinetics-400

8-node 8-A100-GPU training of `MobileViT Spatio-temporal` model can be done using below command:


``` 
# NODE-0
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 0 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-1
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 8 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-2
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 16 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-3
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 24 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-4
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 32 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-5
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 40 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-6
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 48 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

# Node-7
export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc video_classification_results--ddp.rank 56 --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT' --ddp.world-size 64 --ddp.backend nccl

```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluation

To evaluation on the kinetics-400 validation set, run the following code:

``` 
 export CFG_FILE="config/video_classification/kinetics/mobilevit_st.yaml"
 export MODEL_WTS="LOCATION_OF_MODEL_WEIGHTS"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CFG_FILE --common.results-loc video_classification_results \
 --common.override-kwargs model.video_classification.inference_mode=true \
 model.video_classification.pretrained=$MODEL_WTS \
 common.inference_modality=video
```
This command is similar to image classification evaluation, with an exception that we enable inference mode (`model.video_classification.inference_mode=true` OR `model.video-classification.inference-mode`) and set the modality to `video` (`common.inference_modality=video` OR `--common.inference-modality video`)

## Citation

``` 
@inproceedings{mehta2022mobilevit,
    title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
    author={Sachin Mehta and Mohammad Rastegari},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=vh-0sUt8HlG}
}
```