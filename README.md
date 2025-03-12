## About
**Mixture of experts (MoE) models** are increasingly dominant in state-of-the-art LLMs, but experimentation remains largely limited to major labs due to high training complexity.

Existing open-source MoE codebases such as [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) or [Megablocks](https://github.com/databricks/megablocks) provide hardware efficiency across massively scaled cross-node training, but are heavily abstracted and constrained to specific architectural designs. This creates significant barriers for teams wanting to iterate rapidly, develop custom architectures, or understand core MoE design principles.

This repository was developed for our team to rapidly iterate on custom MoE designs, as part of our broader research on **developing intrinsically interpretable MoE architectures**. Our work creates MoE variants that transparently expose human-interpretable computational patterns through routing, such as via [cross-layer experts](https://drive.google.com/file/d/1ytFa0Zr2c-IhW7eRXN6X30sDcba3Qr7t/view?usp=sharing) or [hierarchical experts](https://drive.google.com/file/d/1SDFZoZPbbgTPg6KXbShgiidOu7d9C1tv/view?usp=sharing) (see our [research plan](https://drive.google.com/file/d/18ERsEI6nj_GcAxeZ_h7FeY9q8eeeFdES/view?usp=sharing) for a broad overview, or [interepretability analysis code](https://github.com/bongohead/interpretable-moes-analysis/tree/master)).

<p align="center">
    <img src="https://raw.githubusercontent.com/bongohead/interpretable-moes-analysis/refs/heads/master/images/cross-layer-routing-2.png" alt width="400px">
</p>
<p align="center" >
    <em>Expert activation paths in our cross-layer expert architecture</em>
</p>

Beyond our research goals, we've designed this codebase such that it can be utilized as a simple starting framework for ML teams and researchers to **rapidly develop and train MoE LLMs from scratch**. We prioritize code readability, research flexibility and rapid iteration, avoiding abstraction and dependencies when possible. The code structure is optimized for developing MoEs with total parameters in the 0 - 20B parameter range, and supports single-node (possibly multi-GPU) training setups.

The repository includes both standard, self-contained and easily modifiable MoE architectures and training pipelines, as well as our specialized interpretability-focused implementations. We hope this resource proves valuable for teams looking to experiment with custom MoE designs without the constraints of heavily optimized but inflexible frameworks.

Setup instructions are below.

## Initial Development Setup
1. Acquire a server with one or more CUDA-compatible GPUs (e.g., on Runpod).
2. Clone this repo: `git clone https://github.com/bongohead/interpretable-moes`
3. Add credentials to the repo if you don't have git credentials set already.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
4. To setup basics  and install Python packages, run `sh runpod_setup.sh`. Note this also installs Python 3.12 and sets it as the default Python. This is intended for Runpod, so it may need to be modified for your training environment.
5. Run `echo "/workspace/interpretable-moes" > $(python -c "import site; print(site.getsitepackages()[0])")/add_path.pth`, replacing `/workspace/interpretable-moes` with whatever absolute path you're using. This will add this directory to your Python's system paths so that module paths can be loaded with reference to the top-level folder path (e.g. so you can import from `helpers`).
6. Create a `secrets.env` file with `WANDB_API_KEY=...`.

## Creating and Training an MoE
First, create a new branch and switch to it.

Then, download and prepare your training and validation data: 
1. Run `data/download-sft-data.ipynb` to pull [Fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) data for training and validation. This downloads data as JSON shards into `data/raw-data`. Note that in general, 1 sample => ~1k tokens, so that 200k samples/shard leads to 200M tokens/shard, you'll need ~500 shards to train on 100B tokens.
2. Run `data/process-sft-data.ipynb` to preprocess training and validation shards into token IDs. This may be slow, but it can be run asynchronously during downloading (or during model training).

Next, to develop and train a model:
1. Create a new folder in `experiments`.
2. Copy `experiment.ipynb` from `experiments/base-olmoe` into your new folder. 

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

## Major Experiments
[TBD]
- `base-olmoe` 
- `interp-baselines` 
- `cross-layer-experts`
