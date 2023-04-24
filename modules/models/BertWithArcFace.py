from transformers import BertModel
from modules.losses import ArcFaceLoss
import torch.nn

class BertWithArcFace(nn.Module):

    def __init__(self, last_hidden_size = CFG.bert_hidden_size,num_classes=NUM_CLASSES,dropout=0.5, device = CFG.to(device)):
        super(BertWithArcFace, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.last_hidden_size = last_hidden_size
        self.device = device
        self.model = BertModel.from_pretrained("bert-base-uncased").to(CFG.device)
        self.fc1 = nn.Linear(self.model.config.hidden_size, 768)
        self.bn1 = nn.BatchNorm1d(768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(768, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.margin = ArcFaceLoss(self.last_hidden_size, 
                                           num_classes, 
                                           s=30.0, 
                                           m=0.50, 
                                           easy_margin=False)
        self.fc1 = nn.Linear(self.model.config.hidden_size, 768)

    def forward(self, data, labels=None):
        features = self.model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        features = features[0]
        features = features.mean(dim=1)
        # # Pass through linear layers with ReLU activation
        # features = self.fc1(features)
        # Pass through linear layers with ReLU activation, batch norm, and dropout
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        if labels is not None:
          return self.margin(features, labels).to(device)
        return features