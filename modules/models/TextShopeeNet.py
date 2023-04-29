from torch import nn
import torchvision.models as models
from transformers import BertModel, AutoModel


class TextShopeeNet(nn.Module):
    def __init__(self,
                 model_name='bert-base-uncased',
                 fc_dim=512,
                 freeze=True,
                 dropout=0.0):
        super(TextShopeeNet, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        final_in_features = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.relu = nn.ReLU()
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, data):
        return self.extract_feat(data)

    def extract_feat(self, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        features = x[0]
        features = features[:, 0, :]
        features = self.dropout(features)
        features = self.fc(features)
        features = self.bn(features)
        features = self.relu(features)
        return features
