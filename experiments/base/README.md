## Notes
`experiment.ipynb` contains the entire model class and training code for a standard MoE implementation within a single notebook.

This notebook contains end-to-end code for a baseline MoE LLM implementation, including model classes and model training code. A few notes on the baseline architecture implemented there:

- The base architecture in the notebook largely replicates AI2's [OlMoE-1B-7B](https://arxiv.org/pdf/2409.02060), though the code has been significantly de-abstracted to support quick modification and experimentation, and the training code is written from scratch.
- This base architecture is a typical MoE implementation with layer-level MLP experts and top-k routing. It also has built-in support for optional implementations such as:
    - Shared experts
    - Dense layers before MLP layers
    - A [loss-free load balancing](https://arxiv.org/abs/2408.15664) method (which can be used to supplement the standard load balancing loss or replace it entirely)
- The base architecture in the notebook also contains optional implementations of multi-GPUs expert sharding via the `OlmoeMoe` class. The sharding strategy is to keep all dense layers on a single primary GPU, with experts are distributed among one or more GPUs, and no fractional sharding (i.e. a single expert's entire weights are always one. one GPU). Refer to the `OlmoeModel` initialization in the notebook to understand expert allocation.
- The notebook also contains detailed notes recommending which sections should be modified for building a custom MoE architecture.

To test a custom architecture, modify the model classes in the notebook accordingly and run through the notebook, testing that basic training speed/memory usage are acceptable.

Note that the base code supports multi-GPU setups but they must be on a single node. Multinode/multiprocess setups add significantly complexity for expert training and will require careful implementation of distributed communication not supported here.