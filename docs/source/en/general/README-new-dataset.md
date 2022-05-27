# How to Create a New Dataset Type


Each dataset class in CVNet should be registered with `data.dataset.register_dataset`.
You can either create a new dataset class from scratch or extend one of the existing ones.

This class decorator takes allows you to set a `name` and `task` type for the dataset class:
```python
from data.datasets import register_dataset
from data.datasets.dataset_base import BaseImageDataset

@register_dataset("ade20k", "segmentation")
class ADE20KDataset(BaseImageDataset):
    # PyTorch Dataset type.
```

This allows you to specify this dataset in your config file with the following format:
```yaml
dataset:
  name: "ade20k"
  category: "segmentation"
  # Where the data is stored for train/validation (can be different)
  root_train: "/mnt/vision_datasets/ADEChallengeData2016/"
  root_val: "/mnt/vision_datasets/ADEChallengeData2016/"

```
The `name` and `category` refer to the dataset `name` and `task`.
You can optionally specify the data location using `root_train` and `root_val`.
`BaseImageDataset` will choose the correct path based on the `is_training` and `is_evaluation` parameters.


`BaseImageDataset` is the base class for all datasets currently in CVNet. This is currently only a soft requirement.

## Extending an Existing Dataset

Most of the time, there is no need to create a new dataset class from scratch.
Instead, you can simply extend an existing dataset like `ImagenetDataset`.

The `ImagenetDataset` followds the ImageFolder class in `torchvision.datasets.imagenet`. If your data follows the same format
you can extend ImageNet and only change the parts that are needed, such as including your amazing new transforms:

```python
from data.datasets import register_dataset
from data.datasets.classification.imagenet import ImagenetDataset

@register_dataset("my-new-dataset", "classification")
class AmazingDataset(ImagenetDataset):
    def training_transforms(self, size: tuple or int):
        # My amazing new training-time transforms
```

Keep in mind that you should probably change the `root_train` and `root_val` paths to where your data is located:
```yaml
dataset:
  name: "my-new-dataset"
  category: "classification"
  root_train: "<path-to-training-data>"
  root_val: "<path-to-validation-data>"
```