"""
This file contains calculations for expert-specialization metrics.
"""

import glob
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import math

class ContextLabeledDataset(Dataset):
    """
    A dataset that loads a polysemantic token file.
     Ensures the token is recognized as exactly one token by the tokenizer, and that each meaning has at least 20
     occurrences of that token (summed across its text_samples).
    """

    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        """
        Params:
            @file_path: The file path containing YAML files of samples
            @tokenizer: A HF-style tokenizer
            @max_length: The truncation length for examples
        """
        super().__init__()
        self.records = []
          
        with open(file_path, "r", encoding = "utf-8") as f:
            token_info = yaml.safe_load(f)[0]

        self.test_token = token_info['token']
        self.test_meanings = [m['meaning_label'] for m in token_info['meanings']]

        # Check that 'test_token' is recognized by the tokenizer as exactly one token
        encoded_token = tokenizer(self.test_token, add_special_tokens = False)
        if len(encoded_token["input_ids"]) != 1:
            print(f'Warning: {self.test_token} is not tokenized as a single token, skipping')
            return None
        self.test_token_id = encoded_token["input_ids"][0]

        # For each meaning, sum how many occurrences of token_id appear across all text_samples
        all_meanings_ok = True
        for meaning in token_info['meanings']:
            total_occurrences = 0
            for sample_text in meaning['text_samples']:
                encoded_sample = tokenizer(sample_text, add_special_tokens = False, truncation = True, max_length = max_length)
                total_occurrences += encoded_sample["input_ids"].count(self.test_token_id)
            if total_occurrences < 20:
                all_meanings_ok = False
                break
        # If any meaning fails the minimum occurrences, discard this file
        if not all_meanings_ok:
            print(f'Warning: Minimum token count not met for token {self.test_token}')
            return None

        # Tokenize for getting later
        for meaning in token_info['meanings']:
            for sample_text in meaning["text_samples"]:
                tokenizer_out = tokenizer(sample_text, truncation = True, max_length = max_length, padding = "max_length", return_tensors = "pt")
                record = {
                    "text_samples": sample_text,
                    "test_meaning": meaning["meaning_label"],
                    "input_ids": tokenizer_out["input_ids"][0],  
                    "attention_mask": tokenizer_out["attention_mask"][0]
                }
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
    

def get_context_labeled_data(search_path: str, tokenizer, max_length: int = 512, batch_size: int = 16) -> list[dict]:
    """
    Get a dataloader that returns the text examples with assigned token-meaning labels.
     For use in testing for interpretable expert specialization, via `get_ics` and `get_tis`.
    
    Params:
        @search_path: The file serach path containing YAML files of samples
        @tokenizer: A HF-style tokenizer
        @max_length: The truncation length for examples
        @batch_size: The batch size to return for each dataloader iteration
    
    Example:
        context_aware_test_dataset = get_context_aware_test_data("./../../data/contextual-tokens/samples_*.yaml", tokenizer, 512, 64)

    Returns:
        A list of dictionaries, where each dictionary is structured as:
            ```
            {
                "test_token": str, # The actual polysemantic test token
                "test_token_id": int, # The token ID of that polysemantic test token
                "test_meanings": list[str], # A list of 3 test meanings corresponding to that polysemantic test token
                "dl": torch.utils.data.DataLoader # A pytorch dataloader
            }
            ```
        The dataloader is structured such that it returns a dictionary containing B examples, structured like this:
            ```
            {
                "input_ids": a B x N input_ids tensor
                "attention_mask": a B x N attention_mask tensor
                "test_meanings": a B-length length of the assigned test meanings for each example
                
            }
            ```
        A single example may have multiple instances  of the test_token, though within that example, all usages of the test_token will correspond to the 
        test_meaning. For example, suppose that for a test_token of "@", one of the examples is "My email is test@email.com or test2@email.com.", with a 
        test_meaning of "usage_in_email". This indicates that in that example, *all* uses of "@" correspond to a semantic meaning of being used in an email.
    """

    def collate_fn(batch):
        return {
            "input_ids":  torch.stack([item["input_ids"] for item in batch], dim = 0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim = 0),
            "test_meanings": [item['test_meaning'] for item in batch],
            "raw_text_samples": [item['text_samples'] for item in batch]
        }
    
    yaml_files = glob.glob(search_path)
    if len(yaml_files) == 0:
        raise Exception('No files in search path!')
    print(f"{str(len(yaml_files))} files found")

    dls = []
    for file_path in sorted(yaml_files):
        ds = ContextLabeledDataset(file_path = file_path, tokenizer = tokenizer, max_length = max_length)
        if ds is None or len(ds) == 0:
            continue
        dls.append({
            'test_token': ds.test_token,
            'test_token_id': ds.test_token_id,
            'test_meanings': ds.test_meanings,
            'dl': DataLoader(ds, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
        })
    return dls

def get_js_distance(counts_p, counts_q):
    """
    Compute the Jensen-Shannon distance in log base 2 between two expert count distributions
    
    Params:
        @counts_p, counts_q: two distributions p and q, given as {expert1: count_for_expert1, ..} dicts.

    Returns:
        The JS distance, bounded between 0 and 1
    """
    experts_union = set(counts_p.keys()) | set(counts_q.keys()) # Union of all expert IDs
    sum_p = sum(counts_p.values())
    sum_q = sum(counts_q.values())
    if sum_p == 0 or sum_q == 0: # If either distribution has no occurrences, there's no meaningful difference
        return 0.0
    
    p_dict = {}
    q_dict = {}
    for e in experts_union:
        p_dict[e] = counts_p.get(e, 0) / sum_p
        q_dict[e] = counts_q.get(e, 0) / sum_q
    
    # Compute the mixture m(e) = 0.5*(p(e) + q(e))
    m_dict = {}
    for e in experts_union:
        m_dict[e] = 0.5 * (p_dict[e] + q_dict[e])

    # KL(p||m) and KL(q||m) in log base 2
    kl_pm = 0.0
    kl_qm = 0.0
    eps = 1e-12  # small offset to avoid log(0)

    for e in experts_union:
        pe = p_dict[e]
        me = m_dict[e]
        if pe > 0 and me > 0:
            kl_pm += pe * math.log2((pe + eps) / (me + eps))

    for e in experts_union:
        qe = q_dict[e]
        me = m_dict[e]
        if qe > 0 and me > 0:
            kl_qm += qe * math.log2((qe + eps) / (me + eps))

    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    js_dist = math.sqrt(js_div)
    return js_dist

@torch.no_grad()
def get_ics(model, test_token_data_list, main_device, use_topk1):
    """
    Compute the interpretable context specialization (ICS) metric for each test token x layer. This uses JS distance to compare 
     each meaning-specific expert distribution vs. the overall distribution for that token. This is fast but can be memory intensive; 
     use lower batch sizes via the `test_token_data_list` object if there are memory issues.
    
    Params:
        @model: The model, must return `all_topk_experts` which is a tuple of length equal to # layers, where each element of
          the tuple is a BN x topk tensor of selected expert IDs.
        @test_token_data_list: A list of dictionaries of the exact format returned by `get_context_aware_test_data`.
        @main_device: The device to run calculations on.
        @use_topk1: Whether to only test the topk = 1 expert.

    Returns:
        A dict of format:
            {
                test_token1: {0: .52, 1: .34, ...},
                test_token2: {0: .55, 1: .62, ...},
                ...
            },
          where the keys represent layer indices and the values represent context-awareness scores between 0 - 1
    """
    results = {}

    n_layers = model.conf.n_layers
    n_experts = model.conf.n_experts

    for test_item in test_token_data_list:
        test_token = test_item["test_token"]
        test_token_id = test_item["test_token_id"]
        test_meanings = test_item["test_meanings"]
        dl = test_item["dl"]

        meaning_counts_per_layer = [
            [torch.zeros(n_experts, dtype = torch.long, device = 'cpu') for _ in range(len(test_meanings))]
            for _ in range(n_layers)
        ] # meaning_counts_per_layer[l][meaning_idx]"" torch.LongTensor of shape (n_experts,) expert-length count
        # total_counts_per_layer[l]: same shape (n_experts,), expert-length count for baseline usage
        total_counts_per_layer = [torch.zeros(n_experts, dtype = torch.long, device = 'cpu') for _ in range(n_layers)]

        # Map each meaning string to an index
        meaning_to_idx = {m: i for i, m in enumerate(test_meanings)}

        for batch in dl:
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)
            batch_meanings = batch["test_meanings"]
            B, N = input_ids.shape

            outputs = model(input_ids, attention_mask, moe_method = 'forward_slow', use_lflb = False, use_checkpointing = False)
            all_topk_experts = outputs['all_topk_experts']
            if use_topk1 == True:
                all_topk_experts = tuple(x[..., :1] for x in all_topk_experts)

            flat_ids = input_ids.view(-1)  # shape (B*N, ) # Flatten the input IDs for indexing alignment with all_topk_experts
            
            # Convert each example's meaning label to an integer index shape (B,). e.g. meaning_id_array[b] = meaning_idx of that example
            meaning_id_array = []
            for b_idx in range(B):
                m_str = batch_meanings[b_idx]
                meaning_id_array.append(meaning_to_idx[m_str])
            meaning_id_array = torch.tensor(meaning_id_array, device = main_device)  # (B,)
            meaning_id_array = meaning_id_array.unsqueeze(1).repeat(1, N).view(-1)  # Expand to shape (B, N), so each token in that example has the same meaning


            for l in range(n_layers):
                layer_exps = all_topk_experts[l]  # shape (BN, top_k)

                # A) Baseline distribution for the token - we want all rows where (flat_ids == test_token_id)
                base_mask = (flat_ids == test_token_id)
                base_idx = base_mask.nonzero(as_tuple=True)[0]
                if len(base_idx) > 0:
                    base_exps = layer_exps[base_idx, :] # gather => shape (#rows, top_k)
                    base_exps = base_exps.view(-1) # flatten => (#rows * top_k,)
                    hist_base = torch.bincount(base_exps, minlength = n_experts) # bincount => shape (n_experts,)
                    hist_base = hist_base.cpu()
                    total_counts_per_layer[l] += hist_base

                # B) For each meaning m, gather usage mask: (flat_ids == test_token_id) & (meaning_id_array == m)
                for m_idx in range(len(test_meanings)):
                    meaning_mask = (flat_ids == test_token_id) & (meaning_id_array == m_idx)
                    mm_idx = meaning_mask.nonzero(as_tuple=True)[0]
                    if len(mm_idx) > 0:
                        m_exps = layer_exps[mm_idx, :]
                        m_exps = m_exps.view(-1)
                        hist_m = torch.bincount(m_exps, minlength = n_experts)
                        hist_m = hist_m.cpu()
                        meaning_counts_per_layer[l][m_idx] += hist_m

        # Now compute the average JS distance for each layer (comparing each meaning vs. the overall distribution) then averaging
        layer_js_distances = []
        for l in range(n_layers):
            meaning_dists = []
            
            # Convert total_counts -> python dict 
            base_array = total_counts_per_layer[l].numpy()
            dict_base = {}
            for ex_id, c_val in enumerate(base_array):
                if c_val > 0:
                    dict_base[ex_id] = int(c_val)
            
            # Each meaning
            for m_idx in range(len(test_meanings)):
                sense_arr = meaning_counts_per_layer[l][m_idx].numpy()
                dict_sense = {}
                for ex_id, c_val in enumerate(sense_arr):
                    if c_val > 0:
                        dict_sense[ex_id] = int(c_val)
                d_js = get_js_distance(dict_sense, dict_base)
                meaning_dists.append(d_js)

            avg_js = sum(meaning_dists) / len(meaning_dists)
            layer_js_distances.append(avg_js)

        results[test_token] = {layer_idx: val for layer_idx, val in enumerate(layer_js_distances)}

    return results


@torch.no_grad()
def get_tis(model, test_token_data_list, main_device, pad_token_id, use_topk1 = False):
    """
    Computes a token ID specialization (TIS) metric for each test token x layer, using the JS distance b/t: (a) the distribution of 
      experts used for that token and (b) the distribution of experts used for *all* tokens (excluding padding). This is fast 
      but can be memory intensive; use lower batch sizes via the `test_token_data_list` object if there are memory issues.
    
    Params:
        @model: The model, must return `all_topk_experts` which is a tuple of length equal to # layers, where each element of
          the tuple is a BN x topk tensor of selected expert IDs.
        @test_token_data_list: A list of dictionaries of the exact format returned by `get_context_aware_test_data`.
        @pad_token_id: The ID used for padding, which we should exclude from the "global usage" distribution.

    Returns:
        A dict of format:
            {
                test_token1: {0: .52, 1: .34, ...},
                test_token2: {0: .55, 1: .62, ...},
                ...
            },
          where the keys represent layer indices and the values represent token-specialization scores between 0 - 1
    """
    results = {}

    n_layers = model.conf.n_layers
    n_experts = model.conf.n_experts

    for token_item in test_token_data_list:
        token_str = token_item["test_token"]
        token_id = token_item["test_token_id"]
        dl = token_item["dl"]

        # For each layer, we'll track:
        global_counts_per_layer = [torch.zeros(n_experts, dtype = torch.long) for _ in range(n_layers)]
        token_counts_per_layer = [torch.zeros(n_experts, dtype = torch.long) for _ in range(n_layers)]        

        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)

            outputs = model(input_ids, attention_mask, moe_method = 'forward_slow', use_lflb = False, use_checkpointing = False)
            all_topk_experts = outputs['all_topk_experts']
            if use_topk1 == True:
                all_topk_experts = tuple(x[..., :1] for x in all_topk_experts)
            flat_ids = input_ids.view(-1)  # shape B*N

            nonpad_mask = (flat_ids != pad_token_id) # (A) Non-pad
            token_mask = (flat_ids == token_id) # (B) This specific token

            # For each layer, accumulate counts via bincount
            for l in range(n_layers):
                layer_exps = all_topk_experts[l]  # (B*N, topk)
                
                # A) Global usage (non-pad) gather only the rows where nonpad_mask is True
                nonpad_indices = nonpad_mask.nonzero(as_tuple = True)[0]
                if len(nonpad_indices) > 0:
                    nonpad_rows = layer_exps[nonpad_indices, :] # shape (#nonpad, top_k)
                    nonpad_rows = nonpad_rows.view(-1) # flatten => shape (#nonpad*top_k,)
                    hist_global = torch.bincount(nonpad_rows, minlength = n_experts)
                    global_counts_per_layer[l] += hist_global.cpu()

                # B) Token usage (token_mask)
                token_indices = token_mask.nonzero(as_tuple = True)[0]
                if len(token_indices) > 0:
                    token_rows = layer_exps[token_indices, :] # shape (#token, top_k)
                    token_rows = token_rows.view(-1)
                    hist_token = torch.bincount(token_rows, minlength = n_experts)
                    token_counts_per_layer[l] += hist_token.cpu()
                    
        # Now compute the JS distance for each layer, comparing token_expert_counts vs. global_expert_counts
        layer_js_list = []
        for l in range(n_layers):
            dict_global = {}
            dict_token = {}
            global_arr = global_counts_per_layer[l].numpy()
            token_arr = token_counts_per_layer[l].numpy()
            # Build dictionaries for get_js_distance
            for ex_id, count_val in enumerate(global_arr):
                dict_global[ex_id] = int(count_val)
            for ex_id, count_val in enumerate(token_arr):
                dict_token[ex_id] = int(count_val)
            d_js = get_js_distance(dict_token, dict_global)
            layer_js_list.append(d_js)

        results[token_str] = {i: val for i, val in enumerate(layer_js_list)}
        
    return results



@torch.no_grad()
def get_ec(model, test_token_data_list, main_device, use_topk1 = False):
    """
    Computes an expert continuity metric, measuring the fraction of token occurrences
      of 'test_token' that continue to get routed to the same topk experts in the next layer.

    Args:
        model: The model object. Must return `all_topk_experts` in `outputs["all_topk_experts"]`,
               a tuple of length n_layers, each shaped (B*N, top_k).
        test_token_data_list: The same structure as used for TIS or CA, i.e. a list of dicts:
            [{
               "test_token": str,
               "test_token_id": int,
               "dl": torch.utils.data.DataLoader
             }, ...]
        main_device: The device for the forward pass.
        pad_token_id: The ID used for padding, which we skip.

    Returns:
        A dict with format:
           {
             "token_str": {0: continuity_value, 1: continuity_value, ... n_layers-1: 0.0},
             ...
           }
        where each continuity_value is in [0..1].
    """
    n_layers = model.conf.n_layers

    for token_item in test_token_data_list:
        token_str = token_item["test_token"]
        token_id = token_item["test_token_id"]
        dl = token_item["dl"]

        # overlap_count[l] = number of occurrences that keep at least one expert from layer l->l+1
        # total_count[l]   = total number of token occurrences we see for layer l
        overlap_count = [0 for _ in range(n_layers-1)]
        total_count   = [0 for _ in range(n_layers-1)]

        for batch in dl:
            input_ids = batch["input_ids"].to(model.device)       # (B,N)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(input_ids, attention_mask, moe_method = 'forward_slow', use_lflb = False, use_checkpointing = False)
            all_topk_experts = outputs['all_topk_experts']
            if use_topk1 == True:
                all_topk_experts = tuple(x[..., :1] for x in all_topk_experts)
            flat_ids = input_ids.view(-1)  # shape B*N

            valid_mask = (flat_ids == token_id)
            valid_indices = valid_mask.nonzero(as_tuple = True)[0]  # 1D array of row indices

            # 3) For each layer up to n_layers - 2, check overlap with layer + 1
            # We'll gather the topk experts for those valid_indices in layer l and l+1
            for l in range(n_layers - 1):
                exps_l     = all_topk_experts[l]    # (BN, top_k)
                exps_next  = all_topk_experts[l+1]  # (BN, top_k)

                # Gather relevant rows => shape (#valid, top_k)
                exps_l_valid    = exps_l[valid_indices, :]
                exps_next_valid = exps_next[valid_indices, :]

                total_count[l] += exps_l_valid.size(0) # total_count[l] += #valid

                # Now check overlap row by row in a vectorized manner.
                # If top_k = 1, simpler check: just eq
                if exps_l_valid.size(1) == 1:
                    same = (exps_l_valid[:, 0] == exps_next_valid[:, 0]) # shape (#valid,)
                    overlap_count[l] += same.sum().item()
                else:
                    # exps_l_valid, exps_next_valid: both shape (#valid, top_k)
                    # We'll do a broadcast eq => shape (#valid, top_k, top_k).
                    # If ANY of [top_k x top_k] is True => there's intersection.
                    # Then we reduce "any" across dims 1 & 2 => shape (#valid,).
                    # We'll sum up how many are True => that many have overlap.
                    exps_l_3d = exps_l_valid.unsqueeze(2) # shape (#valid, top_k, 1)
                    exps_next_3d = exps_next_valid.unsqueeze(1) # shape (#valid, 1, top_k)
                    eq_matrix = (exps_l_3d == exps_next_3d) # eq => shape (#valid, top_k, top_k)
                    overlap_bool = eq_matrix.any(dim=(1,2))  # overlap => shape (#valid,)
                    overlap_count[l] += overlap_bool.sum().item()

        continuity_dict = {}
        for l in range(n_layers - 1):
            continuity_dict[l] = overlap_count[l] / total_count[l]
            
        continuity_dict[n_layers-1] = 0.0 # define last layer's continuity=0.0

        results[token_str] = continuity_dict
    return results