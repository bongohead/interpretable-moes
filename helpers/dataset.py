from torch.utils.data import Dataset

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