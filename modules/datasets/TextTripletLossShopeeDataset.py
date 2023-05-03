import torch
from torch.utils.data import Dataset,DataLoader
class TextTripletLossShopeeDataset(Dataset):
    def __init__(self, tokenizer,csv):
        super(TextTripletLossShopeeDataset, self).__init__()
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
        
        text = row.anchor
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        anchor_input_ids = text['input_ids'][0]
        anchor_attention_mask = text['attention_mask'][0] 
        
        text = row.positive
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        positive_input_ids = text['input_ids'][0]
        positive_attention_mask = text['attention_mask'][0] 
        
        text = row.negative
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        negative_input_ids = text['input_ids'][0]
        negative_attention_mask = text['attention_mask'][0] 
        
        return anchor_input_ids, anchor_attention_mask, \
                positive_input_ids, positive_attention_mask, \
                negative_input_ids, negative_attention_mask, \
                torch.tensor(row.label_group) 