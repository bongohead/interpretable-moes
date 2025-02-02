## General Principles 
- The default code only works for a single node (one process) with any amount of GPUs attached to that process. Multi-node is not supported as it add significant additional complexity, and for experimental purposes it's unlikely need multiple nodes/processes anyways.
- The suggested workflow has a lot of "copy and pasting" in order to reduce abstraction and maintain ease of rapid experimentation.

## Creating an Experiment
1. Create a new branch and switch to it.
2. Create a new folder in `experiments`.
3. Run `data/get_data.ipynb` to pull input text data for training and validation.
4. Copy `experiment.ipynb` from `experiments/base-olmoe` into your new folder. This has code for a "base" MoE LLM implementation, including model classes (that support training on single or multi-GPU setups) and training code.
  - The base architecture in the notebook exactly replicates [OlMoE-1B-7B](https://arxiv.org/pdf/2409.02060), though the code has been significantly de-abstracted to support quick modification/experimentation.
  - The base architecture in the notebook also contains several different implementations of multi-GPUs expert sharding via the `OlmoeMoe` class. These all assume the sharding strategy is to keep all dense layers on a single primary GPU, but experts are distributed among one or more GPUs, but with no fractional sharding (i.e. a single expert's entire weights are always one. one GPU). Refer to the `OlmoeModel` initialization in the notebook to understand expert allocation.
  - The notebook also contains detailed notes recommending which sections should be modified for MoE experimentation.
5. To test a custom architecture, modify the model classes and run through the notebook, testing that basic training speed/memory usage are acceptable. Run it on a single GPU first for testing.
6. For multi-GPU training, note that the base code supports multi-GPU setups but they must be on a single node. Multinode/multiprocess setups add significantly complexity for expert training and will require careful additional implementation of distributed communication between experts (e.g. `torch.distributed`)
7. You can either train directly in the notebook or copy code from your modified expeirmentation notebook into seperate Python scripts/modules so you can run from cli.
8. After the experimentation is complete, merge it back into master branch.