# How to Create a New Dataset Type


Each dataset class in CVNet should be registered with `data.dataset.DATASET_REGISTRY`.
You can either create a new dataset class from scratch or extend one of the existing ones.

This class decorator takes allows you to set a `name` and `task` type for the dataset class:
```python
from data.datasets import DATASET_REGISTRY
from data.datasets.dataset_base import BaseImageDataset

@DATASET_REGISTRY.register(name="ade20k", type="segmentation")
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


Currently, all datasets in CVNets are subclasses of either `BaseImageDataset` or `BaseVideoDataset`, which are both
subclasses of `BaseDataset`. This is currently only a soft requirement.

## Extending an Existing Dataset

Most of the time, there is no need to create a new dataset class from scratch.
Instead, you can simply extend an existing dataset like `ImagenetDataset`.

The `ImagenetDataset` follows the ImageFolder class in `torchvision.datasets.imagenet`. If your data follows the same format
you can extend ImageNet and only change the parts that are needed, such as including your amazing new transforms:

```python
from data.datasets import DATASET_REGISTRY
from data.datasets.classification.imagenet import ImagenetDataset

@DATASET_REGISTRY.register(name="my-new-dataset", type="classification")
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
