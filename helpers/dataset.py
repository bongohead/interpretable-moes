import torch
from torch.utils.data import Dataset, DataLoader
import json
import multiprocessing
from functools import partial
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class TextDataset(Dataset):
    def __init__(self, tokenizer_output):
        self.input_ids = tokenizer_output['input_ids']
        self.attention_mask = tokenizer_output['attention_mask']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
    
def load_shard_as_dataloader(shard_path, tokenizer, batch_size, seq_len, eos_seperator_id):
    """
    Loads a shard of text samples from `shard_path` and does an on-the-fly "concatenate-then-chunk" to produce a DataLoader of fixed-length sequences.

    Params:
        @shard_path: Path to a JSON file containing a list of text samples.
        @tokenizer: A HF tokenizer or similar with `encode` method.
        @batch_size: Batch size for the DataLoader.
        @seq_len: Sequence length for the output tokens.
        @insert_eos: If True, insert `tokenizer.eos_token_id` between samples.

    Returns:
        DataLoader that yields (input_ids, attention_mask) for each chunk.
    """
    with open(shard_path, "r") as f:
        texts = json.load(f)

    # Chunk them together
    # No truncation per line - just append all the lines together with an eos_seperator_id in between
    big_token_buffer = []
    for line in texts:
        line_toks = tokenizer.encode(line, add_special_tokens = False)
        big_token_buffer.extend(line_toks)
        big_token_buffer.append(eos_seperator_id)
    
    # Split them into examples of seq_length, padding the last example if shorter
    example_ids_list = []
    i = 0
    while i < len(big_token_buffer):
        example = big_token_buffer[i : i + seq_len]
        i += seq_len

        # Pad the end if needed
        if len(example) < seq_len:
            example = example + [tokenizer.pad_token_id] * (seq_len - len(example))

        example_ids_list.append(example)

    input_ids = torch.tensor(example_ids_list, dtype = torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    ds = TextDataset({'input_ids': input_ids, 'attention_mask': attention_mask})
    dl = DataLoader(ds, batch_size = batch_size, shuffle = True)
    return dl

    
def _tokenize_line(line, tokenizer_encode, eos_id):
    """
    Helper function for parallel tokenization. 'tokenizer_encode' is a bound method or partial function that does 
    tokenizer.encode(..., add_special_tokens=False). 
    """
    line_toks = tokenizer_encode(line)
    # Return line tokens plus the EOS separator
    return line_toks + [eos_id]

def load_shard_as_dataloader_mp(shard_path: str, tokenizer,  batch_size: int,  seq_len: int, eos_seperator_id: int):
    """
    Fast multiprocessing version of `load_shard_as_dataloader`, achieves a ~3x speedup (2mins -> 40 secs) on H200.
    """

    # 1) Read the JSON lines
    with open(shard_path, "r") as f:
        texts = json.load(f)  # a list of text samples

    # 2) Parallel tokenization
    num_processes = max(1, multiprocessing.cpu_count() // 2)
    tokenizer_encode = partial(tokenizer.encode, add_special_tokens=False)

    # Create a pool of processes
    with multiprocessing.Pool(processes = num_processes) as pool:
        tokenized_lines = pool.map(partial(_tokenize_line, tokenizer_encode=tokenizer_encode, eos_id=eos_seperator_id), texts)

    # 3) Concatenate everything into one big token buffer
    big_token_buffer = []
    for toks in tokenized_lines:
        big_token_buffer.extend(toks)

    # 4) Chunk into seq_len blocks, padding the last if needed
    example_ids_list = []
    i = 0
    while i < len(big_token_buffer):
        example = big_token_buffer[i : i + seq_len]
        i += seq_len
        if len(example) < seq_len:
            # pad
            example = example + [tokenizer.pad_token_id] * (seq_len - len(example))
        example_ids_list.append(example)

    # 5) Build the final (input_ids, attention_mask) 
    input_ids = torch.tensor(example_ids_list, dtype=torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 6) Create a simple Dataset & DataLoader
    ds = TextDataset({"input_ids": input_ids, "attention_mask": attention_mask})
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dl