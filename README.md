# About
This contains custom architecture and training code for developing intrinsically interpretable mixture-of-expert (MoE) models. See our [research plan](https://drive.google.com/file/d/18ERsEI6nj_GcAxeZ_h7FeY9q8eeeFdES/view?usp=sharing) for a project overview.


# Research Development Setup
1. Acquire a server with a CUDA-compatible GPU on Runpod.
2. Clone this repo: `git clone https://github.com/bongohead/interpretable-moes`
3. Add credentials to the repo if you don't have git credentials set already.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
4. To install necessary packages, run `sh runpod_setup.sh`.
5. Run `echo "/workspace/interpretable-moes" > $(python -c "import site; print(site.getsitepackages()[0])")/add_path.pth`, replacing `/workspace/interpretable-moes` with whatever absolute path you're using. This will add this directory to your Python's system paths so that module paths can be loaded with reference to the top-level folder path (e.g. so you can import from `helpers`).

# Creating an Experiment
1. Create a new branch and switch to it.
2. Create a new folder in `experiments`.
3. Run `data/get_data.ipynb` to pull [Fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) data for training and validation. This pulls text data as JSON shards. Note that in general, 50k samples => ~50M tokens, so at 50k samples/shard, you'll need ~200 shards to train on 10B tokens. 
4. Copy `experiment.ipynb` from `experiments/base-olmoe` into your new folder. This has code for a "base" MoE LLM implementation, including model classes (that support training on single or multi-GPU setups) and training code.
    - The base architecture in the notebook exactly replicates [OlMoE-1B-7B](https://arxiv.org/pdf/2409.02060), though the code has been significantly de-abstracted to support quick modification/experimentation.
    - The base architecture in the notebook also contains several different implementations of multi-GPUs expert sharding via the `OlmoeMoe` class. These all assume the sharding strategy is to keep all dense layers on a single primary GPU, but experts are distributed among one or more GPUs, but with no fractional sharding (i.e. a single expert's entire weights are always one. one GPU). Refer to the `OlmoeModel` initialization in the notebook to understand expert allocation.
    - The notebook also contains detailed notes recommending which sections should be modified for MoE experimentation.
5. To test a custom architecture, modify the model classes and run through the notebook, testing that basic training speed/memory usage are acceptable. Run it on a single GPU first for testing.
6. For multi-GPU training, note that the base code supports multi-GPU setups but they must be on a single node. Multinode/multiprocess setups add significantly complexity for expert training and will require careful additional implementation of distributed communication between experts (e.g. `torch.distributed`)
7. You can either train directly in the notebook or copy code from your modified expeirmentation notebook into seperate Python scripts/modules so you can run from cli.
8. After the experimentation is complete, merge it back into master branch.