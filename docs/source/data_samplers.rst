=============
Data Samplers
=============

 
CVNet offer data samplers with three sampling strategies:

1. Single-scale with fixed batch size (SSc-FBS)
2. Multi-scale with fixed batch size (MSc-FBS)
3. Multi-scale with variable batch size (MSc-VBS)

For details about these samplers, please see `MobileViT <https://arxiv.org/abs/2110.02178>`_ paper.

Single-scale with fixed batch size (SSc-FBS)
=======

This method is the default sampling strategy in most of the deep learning frameworks (e.g., PyTorch, Tensorflow, and MixNet)
and libraries built on top of them (e.g., the *timm* library).
At the :math:`t`-th training iteration, this method samples a batch of :math:`b` images per GPU [1]_
with a pre-defined spatial resolution of height :math:`H` and width :math:`W`.


Multi-scale with fixed batch size (MSc-FBS)
=======

The SSc-FBS method allows a network to learn representations at a single scale (or resolution).
However, objects in the real-world are composed at different scales. To allow a network to learn representations at multiple scales, MSc-FBS extends SSc-FBS to multiple scales.
Unlike the SSc-FBS method that takes a pre-defined spatial resolution as an input, this method takes a sorted set of $n$ spatial resolutions
:math:`\mathcal{S} = \{ (H_1, W_1), (H_2, W_2), \cdots, (H_n, W_n)\}` as an input.
At the :math:`t`-th iteration, this method randomly samples :math:`b` images per GPU of spatial resolution :math:`(H_t, W_t) \in \mathcal{S}`.


Multi-scale with variable batch size (MSc-VBS):
=======

Networks trained using the MSc-FBS methods are more robust to scale changes as compared to SSc-FBS.
However, depending on the maximum spatial resolution in :math:`\mathcal{S}`, MSc-FBS methods may have a higher peak GPU memory
utilization (see Figure \ref{fig:sampler_perf_cost}) as compared to SSc-FBS; causing out-of-memory errors on GPUs with limited memory.
For example, MSc-FBS with :math:`\mathcal{S} = \{ (128, 128), (192, 192), (224, 224), (320, 320)\}` and :math:`b=256` would need about :math:`2\times` more GPU memory (for images only)
than SSc-FBS with a spatial resolution of :math:`(224, 224)` and :math:`b=256`. To address this memory issue, we extend MSc-FBS to variably-batch sizes.
For a given sorted set of spatial resolutions :math:`\mathcal{S} = \{ (H_1, W_1), (H_2, W_2), \cdots, (H_n, W_n)\}` and a batch size :math:`b`
for a maximum spatial resolution of :math:`(H_n, W_n)`, a spatial resolution :math:`(H_t, W_t) \in \mathcal{S}` with a batch size of
:math:`b_t = \frac{H_n W_n b}{H_t W_t}` is sampled randomly at :math:`t`-th training iteration on each GPU.


Variably-sized video sampler
------

These samplers can be easily extended for videos also.
CVNet provides variably-sized sampler for videos, wherein researchers can control different video-related input
variables (e.g., number of frames, number of clips per video, and video spatial resolution) for learning space- and time-invariant representations.



Data Sampler Objects
====================

.. automodule:: data.sampler.batch_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: data.sampler.multi_scale_sampler
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: data.sampler.variable_batch_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: data.sampler.video_variable_seq_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. [1] The effective batch size is the number of images per GPU times the number of GPUs.
