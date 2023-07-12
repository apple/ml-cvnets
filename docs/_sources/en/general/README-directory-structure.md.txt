# Directory Structure

Here is a quick walk-through of the CVNet code base. Below is a redacted version of the source code directory structure with the important parts marked with a star.

```
.
├── common
├── config
├── cvnets
│   ├── anchor_generator
│   ├── layers
│   ├── misc
│   ├── models  (*)
│   │   ├── classification
│   │   ├── detection
│   │   ├── neural_augmentation
│   │   ├── segmentation
│   │   └── video_classification
│   └── modules  (*)
├── data
│   ├── collate_fns
│   ├── datasets  (*)
│   ├── loader
│   ├── sampler  (*)
│   ├── transforms
│   └── video_reader
├── docs
├── engine  (*)
│   ├── detection_utils
│   └── segmentation_utils
├── loss_fn
├── metrics
└── optim
    └── scheduler
```

### Models
Models for the different tasks (classification, detection, etc) are defined under `cvnets/models/<task>` and are categorized based on the task.
Each task has its own parent class. For example, classification models are derived from the `cvnets.models.classification.base_cls.BaseEncoder` class.

The models are defined to be reusable and shareable between tasks. For example, the ResNet model is defined under the `classification` directory, but the detection models, such as `ssd`, can use ResNet as their `encoder` to avoid duplication.

### Modules
High-level modules such as `InvertedResidual` block for MobileNet-v2 and `TransformerEncoder` block of ViT and MobileViT are available under `cvnets/modules`.

### Datasets
You can find the dataset classes such as `ImagenetDataset` under `data/datasets/<task>` directory. Just like models, dataset are also categorized based on the task.
You can also find the availabe transforms under `data/transforms`. For images, we have transforms based on both OpenCV and PIL, but we recommend using PIL for better performance.

Our novel data samplers can also be found under `data/sampler`.

### Training/Evaluation Engine
The entry scripts (`main_train.py` and `main_eval.py`) will build and initialize objects such as the model, dataset, and optimizer. They also initialize distributed training if need be.

Next, the Training/Evaluation Engine (`engine/training_engine.py` and `engine/evaluation_engine.py`) are called which contain the training/evaluation logic.
