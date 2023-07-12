.. ml-cvnets documentation master file, created by
   sphinx-quickstart on Mon Dec  6 10:41:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVNets documentation!
=====================================

CVNets is an open-source library for training deep neural networks for visual recognition tasks,
including classification, detection, and segmentation.

CVNets supports image and video understanding tools, including data loading, data transformations, novel data sampling methods,
and implementations of several state-of-the-art networks.

Our source code is available on `Github <https://github.com/apple/ml-cvnets>`_ .


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   getting_started
   sample_recipes
   how_to
   data_samplers
   en/general/README-model-zoo
   

Citation
========

If you find CVNets useful, please cite the following papers:

.. code-block::

    @inproceedings{mehta2022mobilevit,
        title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
        author={Sachin Mehta and Mohammad Rastegari},
        booktitle={International Conference on Learning Representations},
        year={2022}
    }

    @inproceedings{mehta2022cvnets, 
        author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
        title = {CVNets: High Performance Library for Computer Vision}, 
        year = {2022}, 
        booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
        series = {MM '22} 
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`
