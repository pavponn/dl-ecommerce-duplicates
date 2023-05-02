import torch
from torch.utils.data import Dataset,DataLoader
class TextArcFaceLossShopeeDataset(Dataset):
    def __init__(self, tokenizer,csv):
        super(TextArcFaceLossShopeeDataset, self).__init__()
        self.csv = csv.reset_index()
        self.tokenizer = tokenizer

    def __len__(self):
        return self.csv.shape[0]
    
    def __get__ids__mask(text):        
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0] 
        
        return input_ids, attention_mask

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        text = row.title
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0] 
        
        return input_ids, attention_mask, torch.tensor(row.label_group) 