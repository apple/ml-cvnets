#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


def train_val_datasets(opts):
    dataset_category = getattr(opts, "dataset.category", None)
    dataset_name = getattr(opts, "dataset.name", None)

    assert dataset_category is not None
    assert dataset_name is not None

    # we may not have access to the dataset, so for CI/CD, we only compute loss
    setattr(opts, "stats.train", "loss")
    # relaxing val statistics to test different metrics
    # setattr(opts, "stats.val", "loss")
    setattr(opts, "stats.checkpoint_metric", "loss")
    setattr(opts, "stats.checkpoint_metric_max", False)

    if dataset_category == "classification":
        # image classification
        from tests.dummy_datasets.classification import (
            DummyClassificationDataset as dataset_cls,
        )
    elif dataset_category == "detection" and dataset_name.find("ssd") > -1:
        # Object detection using SSD
        from tests.dummy_datasets.ssd_detection import (
            DummySSDDetectionDataset as dataset_cls,
        )
    elif dataset_category == "segmentation":
        from tests.dummy_datasets.segmentation import (
            DummySegmentationDataset as dataset_cls,
        )
    elif dataset_category == "video_classification":
        from tests.dummy_datasets.video_classification import (
            DummyVideoClassificationDataset as dataset_cls,
        )
    elif dataset_category == "multi_modal_image_text":
        from tests.dummy_datasets.multi_modal_img_text import (
            DummyMultiModalImageTextDataset as dataset_cls,
        )
    else:
        raise NotImplementedError(
            "Dummy datasets for {} not yet implemented".format(dataset_category)
        )

    train_dataset = dataset_cls(opts)
    valid_dataset = dataset_cls(opts)

    return train_dataset, valid_dataset
