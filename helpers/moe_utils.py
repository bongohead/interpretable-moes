import torch
import math


""" 
These is a dump of helper functions called by the model layers, needed to make forward/backward passes correctly.
- `prepare_4d_causal_attention_mask_with_cache_position` is used to create the upper-triangular infinity mask for attention (not used by flash attention).
- `load_balancing_loss_func` is the usual load balancing function.
- Add any new functions here if needed, but most experiments won't need to touch this section.
"""

# Create the upper-trangular matrix of infinities to mask future tokens in the attention softmax (needed for SDPA + normal attention)
# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L1099C1-L1152 
def prepare_4d_causal_attention_mask_with_cache_position(attention_mask: torch.Tensor, sequence_length: int, target_length: int, dtype: torch.dtype, device: torch.device, cache_position: torch.Tensor, batch_size: int):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value = min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask

# Load balancing loss, copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask):
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim = 0)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (attention_mask[None, :, :, None, None].expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts)).reshape(-1, top_k, num_experts).to(compute_device))
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (attention_mask[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(compute_device))
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def check_cosine_similarity(weights):
    """
    weights: weight matrix on one router: [n_experts, D]
    """    
    weights_normalized = torch.nn.functional.normalize(weights, p=2, dim=1).to(dtype=torch.float32)
    cos_sim = torch.matmul(weights_normalized, weights_normalized.t())
    return cos_sim

def cos_loss_func(model, model_conf):
    sim_matrices = [check_cosine_similarity(model.layers[i].moe.gate.weight) for i in range(model_conf.n_layers)]
    identity_matrix = torch.eye(model_conf.n_experts).to(model_conf.main_device)
    distances = [
        torch.norm(sim_matrix - identity_matrix, p='fro') / model_conf.n_experts * math.sqrt(model_conf.D)   # Normalize by number of experts and sqrt of hidden dimension
        for sim_matrix in sim_matrices
        ]
    
    mean_distance = torch.mean(torch.stack(distances))
    return mean_distance, distances