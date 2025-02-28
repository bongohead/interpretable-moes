import os

import torch
from torch import nn
import torch.nn.functional as F
from dotenv import load_dotenv
import wandb
import math
from helpers.memory import check_memory, profile_memory
from helpers.logging import get_gradient_stats, get_cosine_similarity
from helpers.dataset import load_shard_as_dataloader_mp
from helpers.moe_utils import cos_loss_func
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import glob 
import json
from datetime import datetime

""" 
To do: need to add cos loss to the validation metrics
Now, define a function for calculating validation metrics. This will be later used in the training loop.
- An untrained model should typically return validation loss of ~10 for the base cross-entropy loss.
"""
@torch.no_grad()
def get_val_stats(model, val_dl, router_aux_loss_coef, model_conf):
    """
    Get eval set metrics
    """
    model.eval()

    val_loss_sum = 0.0
    val_base_sum = 0.0
    val_aux_sum = 0.0
    val_steps = 0

    for val_batch in val_dl:
        val_input_ids = val_batch['input_ids'].to(model_conf.main_device)
        val_attn_mask = val_batch['attention_mask'].to(model_conf.main_device)
        
        test_outputs = model(val_input_ids, val_attn_mask, moe_method = 'forward_slow', use_checkpointing = False)

        val_base_sum += test_outputs['base_loss'].detach().cpu().item()
        val_aux_sum  += test_outputs['aux_loss'].detach().cpu().item()
        
        val_loss_sum += (test_outputs['base_loss'] + router_aux_loss_coef * test_outputs['aux_loss']).detach().cpu().item()
        
        val_steps += 1

    avg_test_loss = val_loss_sum / val_steps
    avg_test_base = val_base_sum / val_steps
    avg_test_aux  = val_aux_sum  / val_steps

    model.train()
    return {
        "loss": avg_test_loss,
        "base_loss": avg_test_base,
        "aux_loss": avg_test_aux
    }


def train(model, tokenizer, train_conf, model_conf, val_dl, seed, save_dir):
    """
    Let's train the model.
    - The training loop will loop through training data shards. Each shard will be loaded and concatenated into chunks of size seq_len.
    - Things to consider implementing in the future: more aggressive router LR decay (to encourage router stability)
    """
    # Initialize optimizer/scheduler. The scheduler combines a warmup + cosine annealing.
    optimizer = torch.optim.AdamW(model.parameters(), lr = train_conf.lr, fused = True)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.2, end_factor = 1.0, total_iters = train_conf.warmup_steps), # Warmup
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_conf.decay_steps, eta_min = train_conf.min_lr, last_epoch = -1) # Cosine annealing
        ],
        milestones = [train_conf.warmup_steps]
    )

    # Look for all training data files
    shard_files = sorted(glob.glob("./../../data/train_shard_*.json"))
    print(f"Found {len(shard_files)} shards.")

    # Initialize step count
    step = 0
    total_tokens_trained = 0
    model.train()
    torch.manual_seed(seed)

    for shard_idx, shard_path in enumerate(shard_files):

        print(f"\n=== Loading shard {shard_path} (index {shard_idx}) ===")
        shard_dl = load_shard_as_dataloader_mp(shard_path, tokenizer, batch_size = train_conf.micro_batch_size * train_conf.accumulation_steps, seq_len = train_conf.seq_len, eos_seperator_id = tokenizer.eos_token_id)

        for batch_idx, batch in enumerate(shard_dl):

            # ====================== SPLIT BATCH INTO MICRO-BATCHES ======================
            input_ids = batch['input_ids'].to(model_conf.main_device)
            attention_mask = batch['attention_mask'].to(model_conf.main_device)

            if input_ids.size(0) < (train_conf.accumulation_steps * train_conf.micro_batch_size):
                print(f"Skipping leftover batch, need at least {train_conf.accumulation_steps * train_conf.micro_batch_size}")
                continue

            sub_input_ids = input_ids.split(train_conf.micro_batch_size, dim = 0) 
            sub_attn_mask = attention_mask.split(train_conf.micro_batch_size, dim = 0)

            # ====================== ZERO GRAD ONCE PER "BIG BATCH" ======================
            optimizer.zero_grad()
                    
            # We'll track times and losses across micro-batches
            total_fwd_time = 0.0
            total_bwd_time = 0.0
            total_loss_val = 0.0
            start_batch = time.time()

            # We'll keep a list of dictionaries, one per layer, each mapping expert_id -> usage_count
            usage_accum = [defaultdict(int) for _ in range(model.conf.n_layers)]

            # ====================== MICRO-BATCH LOOP ======================
            for i in range(train_conf.accumulation_steps):

                mb_input_ids = sub_input_ids[i]
                mb_attn_mask = sub_attn_mask[i]

                # ---------------------- Forward ----------------------
                start_fwd = time.time()
                outputs = model(mb_input_ids, mb_attn_mask, moe_method = 'forward_slow', use_lflb = train_conf.use_lflb, use_checkpointing = True)

                loss = outputs['base_loss'] + train_conf.router_aux_loss_coef * outputs['aux_loss']
                fwd_time = time.time() - start_fwd
                total_fwd_time += fwd_time

                # ---------------------- Collect Expert Usage for This Micro-Batch ----------------------
                with torch.no_grad():
                    all_topk_experts = outputs['all_topk_experts']
                    for layer_idx, topk_expert_tensor in enumerate(outputs["all_topk_experts"]):
                        flat_experts = topk_expert_tensor.view(-1)  
                        unique_experts = flat_experts.unique()
                        for ex_id in unique_experts:
                            ex_count = (flat_experts == ex_id).sum().item()
                            usage_accum[layer_idx][int(ex_id)] += ex_count
                            
                # ---------------------- Backward ----------------------
                # Divide by accumulation_steps so total gradient matches "big batch" size
                scaled_loss = loss / train_conf.accumulation_steps
                start_bwd = time.time()
                scaled_loss.backward()
                bwd_time = time.time() - start_bwd
                total_bwd_time += bwd_time

                total_loss_val += loss.item()
            

            # ====================== GRAD CLIPPING & OPT STEP ======================
            shared_params = [p for n,p in model.named_parameters() if 'expert' not in n]
            expert_params = [p for n,p in model.named_parameters() if 'expert' in n]

            torch.nn.utils.clip_grad_norm_(shared_params, train_conf.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(expert_params, train_conf.max_expert_grad_norm)
            
            optimizer.step()
            scheduler.step()

            # ====================== LOSS-FREE BIAS UPDATE ======================
            # We'll do sign-based bias updates after each "big batch"
            if train_conf.use_lflb:
                for layer_ix in range(model.conf.n_layers):
                    model.layers[layer_ix].moe.update_expert_biases(usage_accum[layer_ix], train_conf.bias_update_rate)

            # ============== METRICS ==============
            avg_loss = total_loss_val / train_conf.accumulation_steps # Take the average loss over micro-batches. total_loss_val is the sum of 'loss.item()'.
            total_tokens_trained += attention_mask.sum().detach().cpu().item()
            metrics = {
                'step': step,
                'shard_idx': shard_idx,
                'batch_size': input_ids.shape[0],
                'total_tokens_trained': total_tokens_trained,
                'lr': optimizer.param_groups[0]['lr'],
                'aux_coef': train_conf.router_aux_loss_coef,
                'cos_coef': train_conf.router_cos_loss_coef,
                'train': {
                    'loss': avg_loss,
                    'base_loss': outputs['base_loss'].detach().cpu().item(), # From last microbatch only
                    'aux_loss':  outputs['aux_loss'].detach().cpu().item(), # From last microbatch only
                },
                'fwd_time':  total_fwd_time,
                'bwd_time':  total_bwd_time,
                'batch_time':  time.time() - start_batch
            }

            # ============== EXPENSIVE METRICS (EVERY 10 STEPS) ==============
            if step % 10 == 0:
                
                # Convert usage_accum (list of defaultdicts) into a more standard dict for logging
                usage_dict_final = {}
                for layer_idx, ex_dict in enumerate(usage_accum):
                    usage_dict_final[layer_idx] = dict(ex_dict)  # convert defaultdict -> normal dict
                metrics['expert_usage'] = usage_dict_final
                
                metrics['gradients'] = get_gradient_stats(model)

            # ============== EXTRA EXPENSIVE METRICS (EVERY 500 STEPS) ==============
            if step % 250 == 0:
                metrics['val'] = get_val_stats(model, val_dl, train_conf.router_aux_loss_coef, model_conf)

            # ============== SAVE (EVERY 5000 STEPS) ==============
            if step % 2500 == 0:
                if not os.path.exists(f"saves/{save_dir}"):
                    os.makedirs(f"saves/{save_dir}")
                torch.save(
                    {
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'total_tokens_trained': total_tokens_trained,
                        'step': step
                    },
                    f"saves/{save_dir}/checkpoint_{step:08d}.pt"
                )

            # ============== LOG TO W&B ==============
            wandb.log(metrics)

            # ============== PRINT ==============
            if step <= 10 or (step <= 100 and step % 10 == 0) or (step > 100 and step % 100 == 0):
                print(f"Step {step}: avg_loss={metrics['train']['loss']:.4f} "
                    f"| fwd_time={metrics['fwd_time']:.2f}s | bwd_time={metrics['bwd_time']:.2f}s | batch_time = {metrics['batch_time']:.2f} "
                    f"| lr={metrics['lr']:.1e}"
                )
                
            step += 1
            
            return 
