import torch
import torch.nn.functional as F
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


def gap_loss_func(gate_logits, top_k, attention_mask, upper_bound):
    """
    Compute the gap loss for the gate logits.
    shape of gate_logits[i]: [batch_size * sequence_length, n_experts]
    """
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    
    topk_vals, _ = torch.topk(routing_weights, k=top_k+1, dim=-1) # ensure topk < n_experts 
    k_plus_1_vals = topk_vals[..., -2] - topk_vals[..., -1]  
    
    gap_loss = -torch.log(torch.clamp(k_plus_1_vals, max = upper_bound) + 1e-8)
    
    if attention_mask is not None:
        
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        loss_mask = attention_mask[None, :, :].expand((num_hidden_layers, batch_size, sequence_length)).reshape(-1).to(compute_device)
        gap_loss = gap_loss * loss_mask
        valid_tokens = torch.sum(loss_mask)
    else:
        valid_tokens = gap_loss.numel()
    
    
    total_loss = torch.sum(gap_loss) / valid_tokens
    return total_loss
    
def check_cosine_similarity(weights):
    """
    weights: weight matrix on one router: [n_experts, dimension of the weight]
    """    
    weights_normalized = torch.nn.functional.normalize(weights, p=2, dim=1).to(dtype=torch.float32)
    cos_sim = torch.matmul(weights_normalized, weights_normalized.t())
    return cos_sim

def check_cosine_similarity_per_token(activations):
    """
    activations: [n_tokens, top_k, dimension of the weight]
    """    
    activations_normalized = torch.nn.functional.normalize(activations, p=2, dim=-1).to(dtype=torch.float32)
    cos_sim = torch.matmul(activations_normalized, activations_normalized.transpose(-1, -2))
    return cos_sim


def router_cos_loss_func(model, model_conf):
    sim_matrices = [check_cosine_similarity(model.layers[i].moe.gate.weight) for i in range(model_conf.n_layers)]
    identity_matrix = torch.eye(model_conf.n_experts).to(model_conf.main_device)
    distances = [
        torch.norm(sim_matrix - identity_matrix, p='fro') / model_conf.n_experts * math.sqrt(model_conf.D)   # Normalize by number of experts and sqrt of hidden dimension
        for sim_matrix in sim_matrices
        ]
    
    mean_distance = torch.mean(torch.stack(distances))
    return mean_distance, distances


def expert_cos_loss_func(model, model_conf):
    expert_weights_per_layer = []
    for layer in model.layers:
        layer_weights = []
        for expert in layer.moe.experts:
            layer_weights.append(expert.return_flattened_weights())
        expert_weights_per_layer.append(torch.stack(layer_weights)) # each element in the list is a tensor of shape [n_experts, D]
    
    sim_matrices = [check_cosine_similarity(layer_weights) for layer_weights in expert_weights_per_layer]
    
    identity_matrix = torch.eye(model_conf.n_experts).to(model_conf.main_device)

    distances = [
        torch.norm(sim_matrix - identity_matrix, p='fro') / (model_conf.n_experts) * math.sqrt(model_conf.D * model_conf.I * 3) # ignore the normalization factor for dimension
        for sim_matrix in sim_matrices
    ]
    mean_distance = torch.mean(torch.stack(distances))
    return mean_distance, distances

def differentiable_gram_schmidt(weights, use_random_order=True, keep_magnitude=True, eps=1e-8):
    """
    Perform a fully differentiable Gram–Schmidt orthogonalization on the rows of a weight matrix.
    
    Args:
        weights (Tensor): A 2D tensor of shape (num_vectors, vector_dim).
        use_random_order (bool): If True, process the vectors in a random order.
                                 Otherwise, process them sequentially.
        keep_magnitude (bool): If True, scale the orthogonalized vector to have the same magnitude as the original.
                               If False, return unit-norm orthogonal vectors.
        eps (float): A small constant to avoid division by zero.
        
    Returns:
        Tensor: A new tensor of the same shape with orthogonalized rows.
    """
    num_vectors, num_features = weights.shape
    assert num_vectors <= num_features, "number of vectors should be less than or equal to the number of features"
    # Choose processing order.
    if use_random_order:
        perm = torch.randperm(num_vectors)
    else:
        perm = torch.arange(num_vectors)
    
    new_weights = torch.empty_like(weights)
    orthonormal_list = []  # To store the processed vectors
    
    # Process each vector in the chosen order.
    for idx in perm:
        original_vector = weights[idx]
        v = original_vector.clone()
        # Subtract projections onto all previously processed vectors.
        for u in orthonormal_list:
            v = v - torch.dot(v, u) * u
        # Compute norm after removing components.
        norm_v = torch.norm(v) + eps
        u_new = v / norm_v
        orthonormal_list.append(u_new)
        if keep_magnitude:
            original_norm = torch.norm(original_vector)
            new_weights[idx] = u_new * original_norm
        else:
            new_weights[idx] = u_new

    return new_weights


def representation_orthogonal_loss_func(hidden_states, conf):
    """
    Compute the representation orthogonal loss for the hidden states.
    """
    
    compute_device = hidden_states[0].device
    total_tokens, top_k, D = hidden_states[0].shape
    identity_matrix = torch.eye(top_k).to(compute_device)
    
    # Compute the loss for each hidden state and store in a list
    losses = [
        torch.mean(
            torch.norm(
                zero_diagonal((check_cosine_similarity_per_token(hidden_state) - identity_matrix)), # ignore the diagonal part
                p='fro', dim=(-2, -1)
            ) / top_k * math.sqrt(D)
        )
        for hidden_state in hidden_states
    ]
    
    # Average the total loss over all layers
    mean_loss = torch.mean(torch.stack(losses))
    return mean_loss, losses
        
def zero_diagonal(tensor):
    """
    Zero out the diagonal of a 3d tensor.
    """
    n_slices, m, _ = tensor.shape
    indices = torch.arange(m, device=tensor.device)
    tensor[:, indices, indices] = 0
    return  tensor


def convolve_two_2d(vector_x: torch.Tensor, vector_y: torch.Tensor) -> torch.Tensor:
    """
    Convolve two 2D tensors row-wise using PyTorch's 1D convolution.
    For each row in vector_x (shape (B, L)), it is convolved with the corresponding row in vector_y.
    The kernel (vector_y) is flipped along its last dimension for true convolution.
    Padding is applied (manually, and asymmetrically if needed) so that the output has the same length L as the input rows,
    even when L is even.
    
    Args:
        vector_x (torch.Tensor): 2D tensor of shape (B, L) representing the input signals.
        vector_y (torch.Tensor): 2D tensor of shape (B, L) representing the convolution kernels.
        
    Returns:
        torch.Tensor: 2D tensor of shape (B, L) resulting from convolving each row of vector_x
                      with the corresponding (flipped) row of vector_y.
    """
    B, L = vector_x.shape
    
    # Treat each row as a channel: shape becomes (1, B, L)
    x = vector_x.unsqueeze(0)
    
    # Flip each row of vector_y and reshape for grouped convolution: (B, 1, L)
    kernel = vector_y.flip(dims=[1]).unsqueeze(1)
    
    # Compute asymmetric padding:
    # Total required padding = L - 1, so that output length = (L + (L-1) - L + 1) = L.
    pad_left = (L - 1) // 2
    pad_right = (L - 1) - pad_left
    
    # Manually pad the last dimension (PyTorch expects padding as (left, right))
    x_padded = F.pad(x, (pad_left, pad_right))
    
    # Perform the grouped convolution with no additional padding.
    conv_result = F.conv1d(x_padded, weight=kernel, padding=0, groups=B)
    
    # Remove the batch dimension; result shape will be (B, L)
    return conv_result.squeeze(0)