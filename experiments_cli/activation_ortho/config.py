from dataclasses import dataclass


@dataclass
class OrthoMappingConf:
    """
    Config settings the orthogonal mapping of the weights
    """

    is_gate_orthogonal_init: bool = False # whether to orthogonalize the initial gate weights
    is_freeze_gate_weights: bool = False # whether to freeze the gate weights
    router_cos_loss_coef: float = 0.01 # (relative weight of cosine similarity loss for the router)
    expert_cos_loss_coef: float = 0.01 # (relative wegith of cosine similarty loss for the experts)
    

@dataclass
class ModelConf:
    """
    General config settings for this MoE
    """
    vocab_size: int = 50304 # Base OlMoE: 50304 (vocab size)
    D: int = 2048 # Base OlMoE: 2048 (hidden state dimension)
    H: int = 16 # Base OlMoE: 16 (number of attention heads)
    I: int = 1024 # Base OlMoE: 1024 (expert MLP dimension)
    n_experts: int = 64 # Base OlMoE: 64 (non-shared experts only)
    n_shared_experts: int = 0 # Base OlMoE: 0 (base OlMoE doesn't support shared experts, but may help with inducing expert specialization - see Deepseek paper)
    top_k: int = 8 # Base OlMoE: 8 
    norm_topk_prob: bool = False # Base OlMoE: false (whether to normalize so that expert weights sum to 1 after topk)
    padding_idx: int = 1 # Base OlMoE: 1 (index where padding gets mapped to)
    n_layers: int = 16 # Base OlMoE: 16 (transformer layers)
    rms_norm_eps: float = 1e-05 # Base OlMoE: 1e-05
    rope_theta: float = 10000.0 # Base OlMoe: 10000.0 (this is something needed for ROPE)
    max_position_embeddings: int = 4096 # Base OlMoE: 4096 (this is something needed for ROPE)
    attn_method: str = 'fa2' # In OlMoE this is chosen automatically, here we explicitly pass it - choose 'normal', 'sdpa', or 'fa2'
    main_device: str = 'cuda:0' # Main device for model


""" 
Set training constants to be used for training later.
- The batch size will be equal to micro_batch_size * accumulation_steps.
"""
@dataclass
class TrainConf:
    router_aux_loss_coef: float = 0.01  # Base OlMoE: 0.01 (relative weight of balancing loss)
    gap_loss_coef: float = 0.01 # (relative weight of gap loss)
    ortho_loss_coef: float = 0.01 # (relative weight of orthogonal loss)
    use_lflb: bool = False # Use loss-free load balancing
    bias_update_rate: float = .002 # Bias update rate for lflb
    lr: float = 0.0005 # The starting LR (after warmup)
    min_lr: float = 5e-5 # The minimum LR
    warmup_steps: int = 500 # How long it takes to warmup to the starting LR
    decay_steps: int = 19500 # How long it takes to decay from the starting LR to the minimum LR
    max_grad_norm: float = 1.0 # Gradient clipping for non-expert grads
    max_expert_grad_norm: float = 1.0 # Gradient clipping for expert grads
    micro_batch_size: int = 64 # Size of a microbatch
    accumulation_steps: int = 4 # Number of microbatches within a batch
    seq_len: int = 1024 # The sequence length (exceeds the current limit set by TRITON_MAX_BLOCK['X'] if using 4096)

