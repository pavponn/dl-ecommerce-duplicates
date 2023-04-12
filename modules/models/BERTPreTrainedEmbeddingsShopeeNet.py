from torch import nn
import torchvision.models as models
from transformers import BertModel

class BERTPreTrainedEmbeddingsShopeeNet(nn.Module):

    def __init__(self):
        super(BERTPreTrainedEmbeddingsShopeeNet, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, data):
        outputs = self.model(**data)
        
        # Extract the last hidden states
        last_hidden_states = outputs.last_hidden_state # (N, K, H) K is sequence length and H is hidden state 
         
        # Compute the mean of the last hidden states for each input sequence
        embeddings = last_hidden_states.mean(dim=1)
        
        return embeddings
