# Training MobileNets on the ImageNet dataset

Single node 4-GPU training of `MobileNetv1-1.0` can be done using below command:

``` 
export CFG_FILE="config/classification/imagenet/mobilenet_v1.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

<details>
<summary>
Single node 8-GPU training of `MobileNetv2-1.0`
</summary>

``` 
export CFG_FILE="config/classification/imagenet/mobilenet_v2.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```
</details>


<details>
<summary>
Single node 8-GPU training of `MobileNetv3-Large`
</summary>

``` 
export CFG_FILE="config/classification/imagenet/mobilenet_v3.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```
</details>

## Citation

   * MobileNetv1
``` 
@article{howard2017mobilenets,
  title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}
```

   * MobileNetv2
``` 
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

   * MobileNetv3
``` 
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1314--1324},
  year={2019}
}
```
