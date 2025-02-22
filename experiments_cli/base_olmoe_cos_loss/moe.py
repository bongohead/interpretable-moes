# Imports
import torch
from torch import nn
import torch.nn.functional as F
from dotenv import load_dotenv
import wandb
import math
from helpers.logging import get_gradient_stats
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import os
import glob 
import json
from datetime import datetime

from helpers.moe_utils import prepare_4d_causal_attention_mask_with_cache_position, load_balancing_loss_func
from config import ModelConf


from transformers.modeling_flash_attention_utils import _flash_attention_forward # Flash attention forward

class OlmoeRMSNorm(nn.Module):
    """
    Apply RMS Norm
    - Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L137-L154
    - This is the only norm used in OlMoE!
      - It's used 4 times per layer (attention key norm, attention query norm, layer residual pre-attention norm, post-attention norm)
      - Also one additional time before the final LM head 
    """
    def __init__(self, D, eps = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class OlmoeRotaryEmbedding(nn.Module):
    """
    Get sin/cos ROPE embeddings
    - Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L161-L219
    - Code has been simplified heavily since we're not using dynamic ROPE scaling
    """
    def __init__(self, conf: ModelConf):
        super().__init__()
        dim = int(conf.D/conf.H)
        inv_freq = 1.0 / (conf.rope_theta ** (torch.arange(0, dim, 2, dtype = torch.int64).float()/dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type = device_type, enabled = False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim = -1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype)

class OlmoeAttention(nn.Module):
    """
    Attention implementation
    - Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L288-L391
    - Simplfied to handle base attention/sdpa/flash attention within this one class
    - Also doesn't support GQA (OlMoE doesn't use anyways)
    """
    def __init__(self, conf: ModelConf):
        super().__init__()
        self.attn_method = conf.attn_method
        self.D = conf.D # Hidden state dim
        self.H = conf.H # Num of attention heads
        self.Dh = int(conf.D/conf.H) # Dimensions per head
        
        # Initialize attention layers - no biases following OlMoE architecture
        self.q_proj = nn.Linear(self.D, self.H * self.Dh, bias = False)
        self.k_proj = nn.Linear(self.D, self.H * self.Dh, bias = False)
        self.v_proj = nn.Linear(self.D, self.H * self.Dh, bias = False)
        self.o_proj = nn.Linear(self.D, self.D, bias = False)
        self.q_norm = OlmoeRMSNorm(self.D, eps = conf.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(self.D, eps = conf.rms_norm_eps)

    # Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L223-L255
    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim = 1):
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.LongTensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
        
        B, N , D = hidden_state.shape

        query_state = self.q_norm(self.q_proj(hidden_state)).view(B, N, self.H, self.Dh).transpose(1, 2) # B x N x 2048
        key_state = self.k_norm(self.k_proj(hidden_state)).view(B, N, self.H, self.Dh).transpose(1, 2) # B x N x 2048
        value_state = self.v_proj(hidden_state).view(B, N, self.H, self.Dh).transpose(1, 2) # B x N x 2048

        cos, sin = position_embeddings
        query_state, key_state = self.apply_rotary_pos_emb(query_state, key_state, cos, sin)
        
        if self.attn_method == 'normal':
            attn_weights = torch.matmul(query_state, key_state.transpose(2, 3))/math.sqrt(self.Dh)  # Should be shape B x H x N x N
            attn_weights = attn_weights + attention_mask # Attention mask is upper triangular of negative infinity
            attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_state.dtype)
            attn_output = torch.matmul(attn_weights, value_state) # B x H x N x D/H
            attn_output = attn_output.transpose(1, 2).contiguous() # Reorder into B x N x H x D/H
            attn_output = attn_output.reshape(B, N, D) # Concatenate vertically back into B x N x D
            
        elif self.attn_method == 'sdpa':
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_state, key_state, value_state,
                attention_mask, dropout_p = 0.0, is_causal = True
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(B, N, D)
            
        elif self.attn_method == 'fa2':
            query_state = query_state.transpose(1, 2)
            key_state = key_state.transpose(1, 2)
            value_state = value_state.transpose(1, 2)
            attn_output = _flash_attention_forward(
                query_state, key_state, value_state,
                attention_mask, N, dropout = 0.0, use_top_left_mask = False, is_causal = True
            )
            attn_output = attn_output.reshape(B, N, D).contiguous()
            
        attn_output = self.o_proj(attn_output)
        return attn_output
    
""" 
Now let's define the MLP layer and the MoE layer.
- The MLP layer is simple; modify as needed.
- However, the MoE layer is much more complex, and this layer will probably need to be modified heavily for most experiments.
  - By default, I've defined three forward methods here. As currently implemented, they all generate IDENTICAL outputs but become increasingly more efficient yet complex.
    - `forward_slow` is the most straightforward implementation (similar to the original OlMoE code). It is also the fastest for single-GPU, limited experts (32 or less) operations.
    - `forward_fast` is faster for large # experts, as it places all the relevant states for a single expert to be continguous in memory. For single GPU, it reaches parity w/forward_slow at ~64 experts.
    - `forward_async` is faster for large GPU counts + large # of experts, as it batches all experts who belong on one device together, and also runs them all asynchronously.
    - For initial testing, it's probably best to modify just `forward_slow`, and only modify the others once you want to run a large-scale training run.
  - Each forward method must return a tuple where the first element is the B x N x D MoE layer output, and the second element is the router logits. 
    - To return more, you'll need to also modify the transformer layer class in the next section.
"""
from transformers.activations import silu

class OlmoeMLP(nn.Module):
    """
    Individual expert MLP
    - Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L258-L272
    """
    def __init__(self, conf: ModelConf):
        super().__init__()
        self.conf = conf
        self.D = conf.D
        self.I = conf.I
        self.gate_proj = nn.Linear(self.D, self.I, bias = False)
        self.up_proj = nn.Linear(self.D, self.I, bias = False)
        self.down_proj = nn.Linear(self.I, self.D, bias = False)
        self.act_fn = silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

class OlmoeMoe(nn.Module):
    """
    Entire MLP layer including router
    - Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L604-L649
    """
    def __init__(self, conf: ModelConf):
        super().__init__()
        self.n_experts = conf.n_experts
        self.top_k = conf.top_k
        self.norm_topk_prob = conf.norm_topk_prob
        self.n_shared_experts = conf.n_shared_experts

        self.gate = nn.Linear(conf.D, self.n_experts, bias = False) # Router
        self.experts = nn.ModuleList([OlmoeMLP(conf) for _ in range(self.n_experts)]) # Create experts using OlmoeMLP
        self.shared_experts = nn.ModuleList([OlmoeMLP(conf) for _ in range(self.n_shared_experts)])
        
        # Loss-free load balancing bias ----------------
        # We store a bias value b_i for each expert i.  These are NOT updated by backprop.
        self.register_buffer("expert_biases", torch.zeros(self.n_experts))

    def forward(self, hidden_state: torch.Tensor, moe_method: str, use_lflb: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method routes to one of several possible other forward methods
        """
        if moe_method == 'forward_slow':
            moe_output, router_logits, topk_expert_ids = self.forward_slow(hidden_state, use_lflb)
        elif moe_method == 'forward_fast':
            moe_output, router_logits, topk_expert_ids = self.forward_fast(hidden_state, use_lflb)
        elif moe_method == 'forward_async':
            moe_output, router_logits, topk_expert_ids = self.forward_async(hidden_state, use_lflb)
        else:
            raise ValueError(f'Method "{moe_method}" not implemented.')
        
        # Add output from shared experts
        shared_total = torch.zeros_like(moe_output)
        for shared_expert in self.shared_experts:
            shared_total += shared_expert(hidden_state)
        
        mlp_output = moe_output + shared_total
        return mlp_output, router_logits, topk_expert_ids
        
    # -------------------- LOSS-FREE BIAS UPDATE FUNCTION --------------------
    @torch.no_grad()
    def update_expert_biases(self, usage_counts: dict[int, int], bias_update_rate: float) -> None:
        """
        Given a dict mapping expert_id -> usage_count for this batch, push biases up/down to move usage closer to uniform.
        """
        total_tokens = sum(usage_counts.values())
        mean_usage = total_tokens / self.n_experts

        for ex_id in range(self.n_experts):
            ex_count = usage_counts.get(ex_id, 0) # usage_counts might not have an entry for an expert if it got 0 tokens
            diff = mean_usage - ex_count # e_i = mean_usage - ex_count

            # Sign-based udpate
            if diff > 0:
                self.expert_biases[ex_id] += bias_update_rate
            elif diff < 0:
                self.expert_biases[ex_id] -= bias_update_rate
                
        # Optionally: clamp the biases to avoid extreme values
        self.expert_biases.clamp_(-5.0, 5.0)

    def forward_slow(self, hidden_state: torch.Tensor, use_lflb: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the more intuitive forward pass which loops through each expert slowly
        """
        B, N, D = hidden_state.shape

        # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        hidden_state = hidden_state.view(B * N, D) # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        
        # 1) ---------------- Compute router logits -> choose topk experts and weights ----------------
        router_logits = self.gate(hidden_state) # (BN, n_experts) - routing probability for each token
        routing_weights = F.softmax(router_logits, dim = 1, dtype = torch.float)  # (BN, n_experts)

        # topk_weights and topk_expert_ids  represents for each token, the top k experts and the corresponding weights/expert indices
        if use_lflb:
            router_logits = router_logits + self.expert_biases.unsqueeze(0)
            _, topk_expert_ids  = torch.topk(router_logits, self.top_k, dim = -1) # (BN, top_k)
            topk_weights = torch.gather(routing_weights, 1, topk_expert_ids ) # (BN, top_k)
        else:
            topk_weights, topk_expert_ids  = torch.topk(routing_weights, self.top_k, dim = -1) # both (BN, top_k)

        topk_weights = topk_weights.to(hidden_state.dtype)
        # Optional renormalization if you want the top-k weights to sum to 1
        if self.norm_topk_prob:
            topk_weights /= (topk_weights.sum(dim = -1, keepdim = True) + 1e-9)

        # 2) ---------------- One hot encode - for each expert, which topk x token is active - e.g. expert_assignment_mask[0, :] will be 0s if the first expert is never chosen ----------------
        expert_assignment_mask = F.one_hot(topk_expert_ids , num_classes = self.n_experts).permute(2, 1, 0) # Creates (N_EXPERTS, TOP_K, BN)
        
        # 3) ---------------- Iterate through all the experts, apply each expert to the tokens where the expert are relevant, multiple output by the weights for the topk/token for that expert ----------------
        # Initialize output buffer
        mlp_output = torch.zeros((B * N, D), dtype = hidden_state.dtype, device = hidden_state.device) # Initialize MLP output - later iterate through experts and sum onto this object
        # Iterate
        for expert_ix, expert in enumerate(self.experts):
            
            # For this expert, gives the (topk, token) coordinates which uses the expert
            topk_slot, token_indices = torch.where(expert_assignment_mask[expert_ix, :])
            if token_indices.numel() == 0:
                continue

            # Gather input tokens for this expert
            tokens_for_expert = hidden_state[token_indices, :] # (num_assigned_tokens, D)
            
            # Move to the expert's device
            expert_device = next(self.experts[expert_ix].parameters()).device # Get the device this expert lives on
            tokens_for_expert = tokens_for_expert.to(expert_device) # Move to expert device

            # Forward through expert
            expert_output = expert(tokens_for_expert)
          
            # For each num_assigned_tokens, multiples it by the corresponding weight in topk_slot fort that token_index
            expert_output = expert_output * topk_weights[token_indices, topk_slot].unsqueeze(1).to(expert_device)

            # Move back to original device and acucmulate into mlp output
            expert_output = expert_output.to(mlp_output.device)
            mlp_output.index_add_(0, token_indices, expert_output.to(hidden_state.dtype))

        mlp_output = mlp_output.reshape(B, N, D) # Convert back from BN x D -> B x N x D
        return mlp_output, router_logits, topk_expert_ids
    

    def forward_fast(self, hidden_state: torch.Tensor, use_lflb: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient MoE routing that batches tokens for each expert in a single pass using gather -> scatter operations. 
        - This will be much faster for a large number of experts, but possibly lower for low expert counts.
        """
        B, N, D = hidden_state.shape

        # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        hidden_state = hidden_state.view(B * N, D) # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        
        # 1) ---------------- Compute router logits -> choose topk experts and weights ----------------
        router_logits = self.gate(hidden_state) # (BN, n_experts) - routing probability for each token
        routing_weights = F.softmax(router_logits, dim = 1, dtype = torch.float)  # (BN, n_experts)

        # topk_weights and topk_expert_ids  represents for each token, the top k experts and the corresponding weights/expert indices
        if use_lflb:
            router_logits = router_logits + self.expert_biases.unsqueeze(0)
            _, topk_expert_ids = torch.topk(router_logits, self.top_k, dim = -1) # (BN, top_k)
            topk_weights = torch.gather(routing_weights, 1, topk_expert_ids) # (BN, top_k)
        else:
            topk_weights, topk_expert_ids = torch.topk(routing_weights, self.top_k, dim = -1) # both (BN, top_k)

        # 2) ---------------- Flatten topk results so we can later gather tokens for each expert cleanly ----------------
        topk_expert_ids = topk_expert_ids.view(-1)
        topk_weights = topk_weights.view(-1)

        # [0..BN-1], repeated top_k times (one for each token-expert pair)
        token_indices_flat = torch.arange(B * N, device = hidden_state.device).unsqueeze(1).expand(B * N, self.top_k).reshape(-1) # shape = (BN * top_k,)

        # 3) ---------------- Sort by expert index so we can process contiguous tokens for each expert ----------------
        sorted_expert_ids, expert_sort_indices = torch.sort(topk_expert_ids)
        sorted_token_indices = token_indices_flat[expert_sort_indices]
        sorted_weights = topk_weights[expert_sort_indices]
        sorted_inputs = hidden_state[sorted_token_indices]  # shape = (BN*top_k, D)

        # 4) ---------------- Walk through sorted_experts to find contiguous segments belonging to each expert. We can use torch.unique_consecutive to figure out segment boundaries ----------------
        # Initialize output buffer
        mlp_output = torch.zeros((B * N, D), dtype = hidden_state.dtype, device = hidden_state.device) # Initialize MLP output - later iterate through experts and sum onto this object
        unique_expert_ids, counts = torch.unique_consecutive(sorted_expert_ids, return_counts = True)
        # Iterate
        start_offset = 0
        for expert_id, count_for_this_expert in zip(unique_expert_ids, counts):
            end_offset = start_offset + count_for_this_expert # The chunk [start_offset : end_offset] corresponds to all tokens for this expert

            # Indices for this chunk
            chunk_token_indices = sorted_token_indices[start_offset:end_offset] # (count, D)
            chunk_weights = sorted_weights[start_offset:end_offset].unsqueeze(1) # (count, 1)
            chunk_inputs = sorted_inputs[start_offset:end_offset] # (count, )

            start_offset = end_offset

            # Move to device
            expert_device = next(self.experts[expert_id].parameters()).device
            chunk_inputs = chunk_inputs.to(expert_device)
            chunk_weights = chunk_weights.to(expert_device)

            # Forward pass through this expert
            expert = self.experts[expert_id]
            expert_output = expert(chunk_inputs) # (count, D)

            # Multiply by the top-k gate weight
            expert_output = expert_output * chunk_weights

            # Bring it back to the main device and scatter-add back to the correct token positions
            expert_output = expert_output.to(mlp_output.device)
            mlp_output.index_add_(0, chunk_token_indices, expert_output.to(hidden_state.dtype))

        mlp_output = mlp_output.reshape(B, N, D) # Convert back from BN x D -> B x N x D
        return mlp_output, router_logits, topk_expert_ids
    

    def forward_async(self, hidden_state: torch.Tensor, use_lflb: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Async MoE forward for optimal multi-GPU speeds that:
          1) Flattens tokens
          2) Does gating + top-k
          3) Sorts by (device, expert)
          4) Groups tokens by device, then sub-groups by expert
          5) Uses asynchronous CUDA streams to overlap transfers & expert compute
          6) Accumulates outputs back to the main device
        This is ~2.5x faster than forward_slow with 64 experts on 4 GPUs
        """
        B, N, D = hidden_state.shape        
        main_device = hidden_state.device
        expert_device_map = [str(next(expert.parameters()).device) for expert in self.experts]

        # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        hidden_state = hidden_state.view(B * N, D) # Flatten out B x N x D to BN x D (flattened token-level reps) to route all tokens seperately
        
        # ---------------- 1) Compute router logits -> choose topk experts and weights ----------------
        router_logits = self.gate(hidden_state) # (BN, n_experts) - routing probability for each token
        routing_weights = F.softmax(router_logits, dim = 1, dtype = torch.float)  # (BN, n_experts)

        # topk_weights and topk_expert_ids  represents for each token, the top k experts and the corresponding weights/expert indices
        if use_lflb:
            router_logits = router_logits + self.expert_biases.unsqueeze(0)
            _, topk_expert_ids = torch.topk(router_logits, self.top_k, dim = -1) # (BN, top_k)
            topk_weights = torch.gather(routing_weights, 1, topk_expert_ids) # (BN, top_k)
        else:
            topk_weights, topk_expert_ids = torch.topk(routing_weights, self.top_k, dim = -1) # both (BN, top_k)

        # ---------------- 2) Flatten topk results so we can later gather tokens for each expert cleanly ----------------
        topk_expert_ids = topk_expert_ids.view(-1)
        topk_weights = topk_weights.view(-1)

        # [0..BN-1], repeated top_k times (one for each token-expert pair)
        token_indices_flat = torch.arange(B * N, device = hidden_state.device).unsqueeze(1).expand(B * N, self.top_k).reshape(-1) # shape = (BN * top_k,)

        # ---------------- 3) Sort by expert index so we can process contiguous tokens for each expert ----------------
        sorted_expert_ids, expert_sort_indices = torch.sort(topk_expert_ids)
        sorted_token_indices = token_indices_flat[expert_sort_indices]
        sorted_weights = topk_weights[expert_sort_indices]
        sorted_inputs = hidden_state[sorted_token_indices]  # shape = (BN*top_k, D)

        # ---------------- 4) Build a dict of (expert -> list of tokens) grouped by device  ---------------
        device_to_chunk = defaultdict(list)

        for i in range(sorted_expert_ids.size(0)):
            ex_id = sorted_expert_ids[i].item() # Which expert
            tok_id = sorted_token_indices[i].item() # The actual token index
            w_val = sorted_weights[i].item() # Probability for that token-expert pair
            inp_vec = sorted_inputs[i] # The input embedding
            ex_dev = expert_device_map[ex_id]  # for example, 'cuda:1'
            device_to_chunk[ex_dev].append((ex_id, tok_id, w_val, inp_vec)) # We store enough info to re-group by expert on the same device

        # ---------------- 5) Create streams for each device used ---------------
        device_streams = {}
        for dev_str in set(expert_device_map):
            device_streams[dev_str] = torch.cuda.Stream(device = dev_str)
        # Store partial outputs in a dict of device -> list of (token_ids, output_tensors)
        device_results = defaultdict(list)

        # ---------------- 6) Dispatch to each device in its stream ---------------
        # We'll do a loop over the devices. For each device, we do sub-grouping by expert ID
        for dev_str, tuple_list in device_to_chunk.items():
            if len(tuple_list) == 0:
                continue

            with torch.cuda.stream(device_streams[dev_str]):
                tuple_list.sort(key=lambda x: x[0])  # Sort by ex_id : each tuple is (expert_id, token_idx, w_val, inp_vec)

                # Sub-group by expert
                idx_start = 0
                while idx_start < len(tuple_list):
                    current_ex = tuple_list[idx_start][0]
                    idx_end = idx_start
                    # Gather all entries with the same expert_id
                    while idx_end < len(tuple_list) and tuple_list[idx_end][0] == current_ex:
                        idx_end += 1
                    sub_chunk = tuple_list[idx_start:idx_end]
                    idx_start = idx_end

                    # Unzip the sub-chunk
                    token_ids = [sc[1] for sc in sub_chunk]
                    w_vals = [sc[2] for sc in sub_chunk]
                    inps = [sc[3] for sc in sub_chunk]

                    # Move to main_device -> dev_str as needed
                    token_ids_t = torch.tensor(token_ids, device = main_device, dtype = torch.long)
                    w_vals_t = torch.tensor(w_vals, device = main_device, dtype = hidden_state.dtype)
                    inps_t = torch.stack(inps, dim=0).to(dev_str, non_blocking = True)
                    w_vals_t = w_vals_t.unsqueeze(1).to(dev_str, non_blocking = True)

                    # Forward pass on the expert (on expert device)
                    chunk_output = self.experts[current_ex](inps_t)
                    # Multiply by gating weights
                    chunk_output = chunk_output * w_vals_t  # gating
                    # Move output back to main device
                    chunk_output = chunk_output.to(main_device, non_blocking=True)
                    # Store for now, and index_add_ after we sync
                    device_results[dev_str].append((token_ids_t, chunk_output))

        # ---------------- 7) Synchronize & gather on main device ----------------
        mlp_output = torch.zeros_like(hidden_state, dtype = hidden_state.dtype)

        for dev_str, stream in device_streams.items():
            # Wait for everything launched in that stream to finish
            with torch.cuda.device(dev_str):
                stream.synchronize()

            # Now we can safely do index_add_ on main_device
            for (tok_ids, out_vecs) in device_results[dev_str]:
                mlp_output.index_add_(0, tok_ids, out_vecs)

        # Reshape to [B, N, D]
        mlp_output = mlp_output.view(B, N, D)
        return mlp_output, router_logits, topk_expert_ids


""" 
Now let's define the transformer block.
- Most likely, there is nothing to change here, unless you need to change the input/outputs from the MoE layer.
- Note that this forward pass is nested within a `custom_forward` call in order to support gradient checkpointing.
"""
class OlmoeBlock(nn.Module):
    """
    A single transformer layer
    """
    def __init__(self, conf: ModelConf, layer_idx: int):
        super().__init__()
        self.D = conf.D
        self.self_attn = OlmoeAttention(conf = conf)
        # self.self_attn = torch.compile(self.self_attn) # attn layers can be compiled for speed 
        self.moe = OlmoeMoe(conf)
        self.input_layernorm = OlmoeRMSNorm(conf.D, eps = conf.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(conf.D, eps = conf.rms_norm_eps)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        moe_method: str,
        use_lflb: bool,
        use_checkpointing: bool
        ):
            
        def custom_forward(hidden_state: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.LongTensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], moe_method: str, use_lflb: bool = False):

            ### Pre-SA Residual Stream + Norm ###
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
            
            ### SA + Sum to Residual Stream ###
            attn_output = self.self_attn(
                hidden_state,
                attention_mask = attention_mask,
                position_ids = position_ids,
                position_embeddings = position_embeddings
            )
            hidden_state = residual + attn_output

            ### Pre-MLP Residual Stream + Norm ###
            residual = hidden_state
            hidden_state = self.post_attention_layernorm(hidden_state)
            
            ### MLP + Sum to Residual Stream###
            mlp_output, router_logits, topk_experts = self.moe(hidden_state, moe_method = moe_method, use_lflb = use_lflb)
            hidden_state = residual + mlp_output
            
            return hidden_state, router_logits, topk_experts
    
        if use_checkpointing:
            # Use gradient checkpointing to reduce activation memory
            hidden_state, router_logits, topk_experts = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_state, attention_mask, position_ids, position_embeddings, moe_method, use_lflb, use_reentrant = True
            )
        else:
            # Normal forward pass
            hidden_state, router_logits, topk_experts = custom_forward(
                hidden_state, attention_mask, position_ids, position_embeddings, moe_method, use_lflb
            )
            
        return hidden_state, router_logits, topk_experts

""" 
Now define the top-level model.
- This class is initialized with the `ModelConf` config settings as well as a list of expert-device mappings (leave blank for single-GPU tests).
- After initialization, it creates all child layers and moves the experts to their correct devices. 
  - All other parameters will continue to exist on the default device.
- Modify `_init_weights` to change the weight initialization scheme.
- The forward pass calls the children layers and also calculates the loss (standard cross-entropy + aux loss). 
- Modifcation from Bo: force orthogonal initialization of gate weights; extract gate weights from each layer's MoE
"""
from transformers.loss.loss_utils import ForCausalLMLoss # Cross-entropy loss that handles label shifting

class OlmoeModel(nn.Module):
    """
    The top level model object. Also handles weight initialization and loss calculations.
    """
    def __init__(self, conf: ModelConf, primary_device: str, expert_device_map: None|list[str] = None):
        """
        Params:
            @conf: A configuration object of class ModelConf.
            @primary_device: A device for which to store the dense layers and shared experts on.
            @expert_device_map: A list of devices to store experts on. If `None`, stores them all on whatever the torch default device is.
              For example, `expert_device_map = ['cuda:0', 'cuda:1', 'cuda:1', 'cuda:2']` means to store expert 0 on cuda:0, experts 1-2 on the device cuda:1, and expert 3 on cuda:2.
        """
        super().__init__()
        self.conf = conf
        
        ### Layers ###
        self.embed_tokens = nn.Embedding(self.conf.vocab_size, self.conf.D, self.conf.padding_idx)
        self.rotary_emb = OlmoeRotaryEmbedding(conf = self.conf)
        self.layers = nn.ModuleList([OlmoeBlock(self.conf, layer_idx) for layer_idx in range(self.conf.n_layers)])
        self.norm = OlmoeRMSNorm(self.conf.D, eps = self.conf.rms_norm_eps)
        self.lm_head = nn.Linear(self.conf.D, self.conf.vocab_size, bias = False)
        
        ### Initialize weights ###
        self.apply(self._init_weights)
        if self.conf.gate_orthogonal:
            for layer in self.layers:
                self._orthogonal_weights(layer.moe.gate.weight) # orthogonal initialization of gate weights
                layer.moe.gate.weight.requires_grad = not self.conf.is_freeze_weights
            
        ### Model ###
        self.to(primary_device)

        ### Experts ###
        if expert_device_map is not None:
            self._move_experts_to_devices(expert_device_map)

    # OlMoE weight initiation - see https://github.com/huggingface/transformers/blob/8f1509a96c96747c893051ac947795cfb0750357/src/transformers/modeling_utils.py#L2500-L2515
    # Normal distribution for linear layers + embeddings
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
            # In the vocab -> embedding layer, set all embeddings to 0 for the padding token (tokenizer.pad_token_id)
            if module is self.embed_tokens:
                self.embed_tokens.weight.data[self.conf.padding_idx].zero_() 
        # Seems to use default weight initialization for other layers
            # Move all parameters and buffers to the specified dtype
            
    def _orthogonal_weights(self, weights):
        """
        weights: weight matrix on one router: [n_experts, D]
        Change the weights in place
        """
        data_type = weights.dtype
        weights.data = nn.init.orthogonal_(weights.to(dtype=torch.float32)).to(dtype=data_type) 
            
    def _move_experts_to_devices(self, expert_device_map: list[str]):
        """
        Move each expert in each layer's MoE to the specified device.
        """
        # Require that the length of expert_device_map equal the length of conf.n_experts.
        n_experts = self.conf.n_experts
        if len(expert_device_map) != n_experts:
            raise ValueError(f"expert_device_map has length {len(expert_device_map)} but n_experts = {n_experts}.")
            
        for _, layer in enumerate(self.layers):
            moe_block = layer.moe 
            for ex_idx, expert in enumerate(moe_block.experts):
                target_dev = expert_device_map[ex_idx]
                expert.to(target_dev)
                
    def _extract_gate_weights(self):
        gate_weights = []
        for layer in self.layers:
            gate_weights.append(layer.moe.gate.weight.data)
        return gate_weights
    
    def _extract_expert_weights(self):
        expert_weights = []
        for layer in self.layers:
            expert_weights.append(layer.moe.experts.weight.data)
        return expert_weights
            
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, moe_method: str, use_lflb: bool = False, use_checkpointing : bool = False):
        """
        Params:
            @input_ids: A tensor of input IDs of size B x N, where B is the batch size and N is the sequence length.
            @attention_mask: An attention mask tensor of size B x N.
            @moe_method: The method to use to calculate the MoE routing. See the `OlmoeMoe` class for details.
            @use_lflb: Whether or not to use loss-free balancing.
            @use_checkpointing: Whether to use gradient checkpointing. Only set `True` during training.
        """
        hidden_state = self.embed_tokens(input_ids)
        B, N, D = hidden_state.shape

        ### Prep rotary embeddings + attention masks  ###
        cache_position = torch.arange(0, N, device = hidden_state.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_state, position_ids) # Position embeddings to be shared across transformer layers

        # This is the upper-trangular matrix of infinities to mask future tokens in the attention softmax;
        if self.conf.attn_method in ['normal', 'sdpa']:
            causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(attention_mask, N, N, hidden_state.dtype, hidden_state.device, cache_position, B)
        # The flash attention mask is simpler - takes only the original attention mask or None
        elif self.conf.attn_method == 'fa2':
            causal_mask  = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        
        ### Transformer layers ###
        all_router_logits = () # Save router logits from each layer into this; will be needed for load balancing loss
        all_topk_experts = () # Return topk experts
        
        for _, layer in enumerate(self.layers):
            hidden_state, router_logits, topk_experts = layer(
                hidden_state,
                attention_mask = causal_mask,
                position_ids = position_ids,
                position_embeddings = position_embeddings,
                moe_method = moe_method,
                use_lflb = use_lflb,
                use_checkpointing = use_checkpointing
            )
            all_router_logits += (router_logits, )
            all_topk_experts += (topk_experts,)  # Store the topk_experts for each layer

        hidden_state = self.norm(hidden_state)
        output_logits = self.lm_head(hidden_state)

        ##### Calculate Loss #####
        # The labels object should be a tensor of token IDs or -100 (for attention mask, since don't want to calculate loss for those)
        label_ids = torch.where(input_ids == self.conf.padding_idx, torch.tensor(-100), input_ids)
        # Get regular loss
        base_loss = ForCausalLMLoss(output_logits, label_ids, self.conf.vocab_size)
        # Get load balancing loss
        aux_loss = load_balancing_loss_func(gate_logits = all_router_logits, num_experts = self.conf.n_experts, top_k = self.conf.top_k, attention_mask = attention_mask)

        return {
            'all_router_logits': all_router_logits,
            'all_topk_experts': all_topk_experts,
            'logits': output_logits,
            'aux_loss': aux_loss,
            'base_loss': base_loss
        }