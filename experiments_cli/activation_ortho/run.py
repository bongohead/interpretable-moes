import torch
from torch import nn
import torch.nn.functional as F
from dotenv import load_dotenv
import wandb
import math
from helpers.memory import check_memory, profile_memory
from helpers.logging import get_gradient_stats
from helpers.moe_utils import check_cosine_similarity
from helpers.dataset import load_shard_as_dataloader_mp
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import os
import glob 
import json
from datetime import datetime
from transformers import AutoTokenizer

from config import ModelConf, TrainConf, OrthoMappingConf
from moe import OlmoeModel
from train import train


RUN_NAME = 'baseline_small'
RUN_NOTES = 'default baseline, without any loss including load-balancing loss'
save_dir = 'baseline_small'

model_conf = ModelConf(
    D = 768, 
    H = 12,
    I = 3072,
    n_experts = 16,
    n_shared_experts = 0,
    top_k = 2,
    norm_topk_prob = False,
    n_layers = 12,
    max_position_embeddings = 1024,
    main_device = 'cuda:0'
)

train_conf = TrainConf(
    micro_batch_size = 64,
    accumulation_steps = 8,
    seq_len = 1024, 
    use_lflb = False,
    gap_loss_coef = 0,
    router_aux_loss_coef = 0,
    ortho_loss_coef = 0.1
)

or_conf = OrthoMappingConf(
    is_gate_orthogonal_init = False,
    is_freeze_gate_weights = False,
    router_cos_loss_coef = 0,
    expert_cos_loss_coef = 0

)

seed = 3456


""" 
Let's load the model
- Set the default_device to specify where all the non-expert layers live (the experts are moved on model init)
- Set the default_dtype to specify the model dtype, all params will be in this dtype except for this explicitly specified differently in class definition
  - In the default OlMoE, RMSNorm is required to be f32 whereas all other params are bf16. 
"""
# torch.set_default_device(conf.main_device) # This is buggy, don't use
torch.set_default_dtype(torch.bfloat16)
torch.set_float32_matmul_precision('medium') # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html 
torch.manual_seed(seed)

model = OlmoeModel(
    model_conf,
    or_conf,
    primary_device = model_conf.main_device, # Where to store dense layers and shared experts
    expert_device_map = [model_conf.main_device] * model_conf.n_experts #=, here let's test them with all of them on cuda:0
)
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained('allenai/OLMoE-1B-7B-0924', add_eos_token = False, add_bos_token = False)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
check_memory()


"""
Setup a Wandb run for logging. Choose a run name and notes for the run!
"""

load_dotenv('./../../secrets.env')
wandb.login(key = os.getenv('WANDB_API_KEY'))
run = wandb.init(
    project = 'interpretable-moes', 
    name = RUN_NAME,
    notes = RUN_NOTES,
    config = {**asdict(model_conf), **asdict(train_conf), **asdict(or_conf)}
)

# (Optional) Also log various info as a wandb media object.
additional_log_notes = {
    'run_name': RUN_NAME,
    'notes': RUN_NOTES,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_model_params': sum(p.numel() for p in model.parameters()),
    'available_cuda_gpus': [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())],
    'model_conf': asdict(model_conf),
    'train_conf': asdict(train_conf),
    'or_conf': asdict(or_conf)
}

wandb.log({'conf': wandb.Html(f"<pre style='font-size:12px;'>{json.dumps(additional_log_notes, indent = 2)}</pre>")})


val_dl = load_shard_as_dataloader_mp(
    './../../data/val_shard.json',
    tokenizer,
    batch_size = 32,
    seq_len = 2048,
    eos_seperator_id = tokenizer.eos_token_id
)

train(model, tokenizer, train_conf, model_conf, or_conf, val_dl, seed, save_dir)
wandb.finish()