import torch 
import numpy as np 

@torch.no_grad()
def get_gradient_stats(model, VANISHING_THRESHOLD: float = 1e-5, EXPLODING_THRESHOLD: float = 10.0):
    """
    Returns a hierarchical dictionary of gradient statistics for a mixture-of-experts model.
        - Tracks global stats (max norm, mean norm, NaN/Inf checks, etc.).
        - Tracks per-layer stats (if parameter names contain "layers.{i}").
        - Distinguishes expert vs. shared parameters based on "expert" in the name.
    
    Params:
        @model: The model object
        @VANISHING_THRESHOLD: The threshold to consider gradients to be vanishing
        @EXPLODING_THRESHOLD: The threshold to consider gradients to be exploding

    Args:
        model: PyTorch model with parameters that may have .grad

    Returns:
        stats (dict): {
            "global": {
                "max_l2_norm": float,
                "mean_l2_norm": float,
                "max_abs_value": float,
                "has_nan": bool,
                "has_inf": bool,
                "pct_vanishing": float,
                "pct_exploding": float
            },
            "layers": {
                "0": {"norms": [...], "max_l2_norm": ..., "mean_l2_norm": ...},
                "1": {...},
                ...
            },
            "experts": {
                "norms": [...],
                "max_norm": float,
                "mean_norm": float
            },
            "shared": {
                "norms": [...],
                "max_norm": float,
                "mean_norm": float
            },
        }
    """
    # Initialize our main stats dictionary
    stats = {
        "global": {
            "max_l2_norm": 0.0,
            "mean_l2_norm": 0.0,
            "max_abs_value": 0.0,
            "has_nan": False,
            "has_inf": False,
            "pct_vanishing": 0.0,
            "pct_exploding": 0.0
        },
        "layers": {},  # layer "0", layer "1", etc.
        "experts": {
            "norms": [],
            "max_norm": 0.0,
            "mean_norm": 0.0
        },
        "shared": {
            "norms": [],
            "max_norm": 0.0,
            "mean_norm": 0.0
        }
    }

    # We'll collect global stats across all parameters:
    grad_norms_all = []
    grad_maxvals_all = []

    # Counters for vanish/explode
    num_params = 0
    num_vanishing = 0
    num_exploding = 0

    # We'll check NaN/Inf in a single pass to avoid repeated CPU transfers
    has_nan = False
    has_inf = False

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Check for NaNs / Infs
        if torch.isnan(param.grad).any():
            has_nan = True
        if torch.isinf(param.grad).any():
            has_inf = True

        grad = param.grad.detach()
        grad_norm = grad.norm(2).item()
        grad_maxv = grad.abs().max().item()

        grad_norms_all.append(grad_norm)
        grad_maxvals_all.append(grad_maxv)

        num_params += 1
        if grad_norm < VANISHING_THRESHOLD:
            num_vanishing += 1
        if grad_norm > EXPLODING_THRESHOLD:
            num_exploding += 1

        # If the parameter name contains "layers.{i}", we treat it as that layer's param
        if "layers." in name:
            layer_idx = name.split("layers.")[1].split(".")[0]  # e.g. "0", "1", ...
            if layer_idx not in stats["layers"]:
                stats["layers"][layer_idx] = {"norms": []}
            stats["layers"][layer_idx]["norms"].append(grad_norm)

        # Distinguish experts vs. shared parameters
        if "expert" in name:
            stats["experts"]["norms"].append(grad_norm)
        else:
            stats["shared"]["norms"].append(grad_norm)

    # ----------------- Fill Global Stats -----------------
    if len(grad_norms_all) > 0:
        stats["global"]["max_l2_norm"] = max(grad_norms_all)
        stats["global"]["mean_l2_norm"] = sum(grad_norms_all) / len(grad_norms_all)
        stats["global"]["max_abs_value"] = max(grad_maxvals_all)
        stats["global"]["pct_vanishing"] = 100.0 * (num_vanishing / num_params)
        stats["global"]["pct_exploding"] = 100.0 * (num_exploding / num_params)

    stats["global"]["has_nan"] = has_nan
    stats["global"]["has_inf"] = has_inf

    # ----------------- Fill Layer Stats -----------------
    # Each layer "i" might have multiple parameters, so we compute max/mean from "norms"
    for layer_idx, layer_dict in stats["layers"].items():
        norms = layer_dict["norms"]
        layer_dict["max_l2_norm"] = float(np.max(norms))
        layer_dict["mean_l2_norm"] = float(np.mean(norms))

    # ----------------- Fill Expert/Shared Stats -----------------
    for group in ["experts", "shared"]:
        norms = stats[group]["norms"]
        if norms:
            stats[group]["max_norm"] = float(np.max(norms))
            stats[group]["mean_norm"] = float(np.mean(norms))

    return stats


@torch.no_grad()
def get_cosine_similarity(model, model_conf):
    """
    Compute the cosine similarity of the gate weights of each layer.
    """
    cos_similarity_matrices = []
    for layer in model.layers:
        cos_similarity_matrices.append(check_cosine_similarity(layer.moe.gate.weight))
    return cos_similarity_matrices