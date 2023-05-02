import torch
import torch.nn as nn
from transformers import BertModel

class BertWithTriplet(nn.Module):
    def __init__(self):
        super(BertWithTriplet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 768)
        
    def forward(self, input_ids, attention_mask):
        # Get the output of BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.fc(outputs.last_hidden_state[:, 0, :])
        return embeddings