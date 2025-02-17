import torch 

@torch.no_grad()
def get_rrs_expert_continuity(all_topk_experts: list[torch.Tensor], attention_mask: torch.Tensor, n_experts: int, max_steps: int = 2):
    """
    Description: 
        Define the cross-expert continuity CE(s) is defined as the aggregate percentage of tokens that remain in the 
        same expert e across layers ll through l+s, averaged (or summed) across all layers l and experts e.

    Params:
        @all_topk_experts: tuple of length K (# of layers), where all_topk_experts[l] is shape (BN, top_k).
        @n_experts: Total number of (non-shared) experts.
        @max_steps: For multi-step continuity, how many consecutive layers to track.
        @attention_mask: The standard B x N attention mask.

    Returns:
        A dict {1: .5, 2: .2, etc} => At any given token at layer l at expert e, 50% of the tokens 
        Note: With a random init and topk=3, n_experts=15, you will typically get a cross_layer_expert_metric[1] of around 35-45%, not 3/15 = 20%. This is because layers will be correlated - hidden states are still correlated across layers, even with randominit!
    """
    L = len(all_topk_experts)
    attention_mask = (attention_mask.view(-1) == 1) # Cast to (BN)
    T = attention_mask.size(0)

    # ----- 1) Build sets of token indices S[l][e] -----
    sets_per_layer = []
    for l in range(L):
        topk_l = all_topk_experts[l]  # shape (T, top_k)
        expert_sets = [set() for _ in range(n_experts)]
        for t in range(T):
            if not attention_mask[t].item():
                continue
            chosen_exps = topk_l[t].tolist()
            for e in chosen_exps:
                expert_sets[e].add(t)
        sets_per_layer.append(expert_sets)

    # ----- 2) Multi-step continuity -----
    # `Get a 3D structure [L][e][step] => At L, fraction a  e at staying at e for up to step layers
    multi_cont = [ [dict() for _ in range(n_experts)] for _ in range(L)]
    for l in range(L):
        for e in range(n_experts):
            base = sets_per_layer[l][e]
            denom = len(base)
            if denom == 0:
                for st in range(1, max_steps+1):
                    multi_cont[l][e][st] = 0.0
                continue
            running = base
            for st in range(1, max_steps+1):
                nxt = l + st
                if nxt >= L:
                    multi_cont[l][e][st] = 0.0
                else:
                    running = running.intersection(sets_per_layer[nxt][e])
                    multi_cont[l][e][st] = len(running) / denom

    # ----- 3) Cross-Layer, Cross-Expert Metric -----
    # metric[st] = fraction of all (layer l, expert e, token t) usage events that remain with the same expert e for st consecutive layers
    # (only counting usage events that can measure st steps).
    consecutive_sequences = [0] * (max_steps + 1)
    matched_sequences = [0] * (max_steps + 1)

    for l in range(L):
        for e in range(n_experts):
            base = sets_per_layer[l][e]
            # For each token t in expert e at layer l
            for t in base:
                # Try st in [1..max_steps]
                for st in range(1, max_steps + 1):
                    # Must have enough layers to measure st steps
                    if l + st >= L:
                        break
                    consecutive_sequences[st] += 1

                    # Check if token t remains in expert e for next st layers
                    keep = True
                    for i in range(1, st + 1):
                        if t not in sets_per_layer[l + i][e]:
                            keep = False
                            break
                    if keep:
                        matched_sequences[st] += 1

    # Convert to fraction
    cross_layer_expert_metric = {}
    for st in range(1, max_steps + 1):
        if consecutive_sequences[st] == 0:
            cross_layer_expert_metric[st] = 0.0
        else:
            cross_layer_expert_metric[st] = (matched_sequences[st] / consecutive_sequences[st])

    # Return everything, including the new metrics
    return cross_layer_expert_metric  # fraction of (l,e) with >=1 token