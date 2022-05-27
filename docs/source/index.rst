.. ml-cvnets documentation master file, created by
   sphinx-quickstart on Mon Dec  6 10:41:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVNet's documentation!
=====================================

CVNet is a high-performance open-source library for training deep neural networks for visual recognition tasks,
including classification, detection, and segmentation.

CVNet supports image and video understanding tools, including data loading, data transformations, novel data sampling methods,
and implementations of several state-of-the-art networks with significantly better performance than the original publications.

.. Our source code is available at: \url{https://github.pie.apple.com/sachin-mehta/ml-cvnets}.


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   getting_started
   sample_recipes
   how_to
   data_samplers
   en/general/README-model-zoo
..    models
   

Citation
========

If you find CVNets useful, please cite the following papers:

.. code-block::

    @article{mehta2021mobilevit,
        title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
        author={Mehta, Sachin and Rastegari, Mohammad},
        journal={arXiv preprint arXiv:2110.02178},
        year={2021}
    }

    @article{mehta2022cvnets,
        title={CVNets: High Performance Library for Computer Vision},
        author={Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
        year={2022}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`
