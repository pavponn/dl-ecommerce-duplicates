import torch
from transformers import BertModel
from modules.losses.BertArcFaceLoss import BertArcFaceLoss
import torch.nn as nn
from modules.cfg.BertArcFaceLossCFG import BertArcFaceLossCFG as CFG
import transformers
from torch.nn import functional as F

class BertWithArcFace(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='bert-base-uncased',
                 pooling='mean_pooling',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 mode_train=True):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(BertWithArcFace, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        final_in_features = self.transformer.config.hidden_size
        
        self.pooling = pooling
        self.use_fc = use_fc
        self.mode_train = mode_train
    
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            # self.relu = nn.ReLU()
            self._init_params()
            final_in_features = fc_dim
        
        # if self.mode_train:
        #     self.loss_module = loss_module
        #     if loss_module == 'arcface':
        #         self.final = BertArcFaceLoss(final_in_features, n_classes,
        #                                       s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        #     else:
        #         self.final = nn.Linear(final_in_features, n_classes,bias=False)

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = BertArcFaceLoss(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(final_in_features, n_classes,bias=False)


    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids,attention_mask, label):
        # feature = self.extract_feat(input_ids,attention_mask)
        # if self.mode_train:
        #     if self.loss_module == 'arcface':
        #         logits = self.final(feature, label)
        #     else:
        #         logits = self.final(feature)
        #     return logits
        # else:
        #     return F.normalize(feature)
        feature = self.extract_feat(input_ids,attention_mask)
        if self.loss_module == 'arcface':
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)

        if self.mode_train:
            return logits
        else:
            return F.normalize(feature),logits
        


    def extract_feat(self, input_ids,attention_mask):
        x = self.transformer(input_ids=input_ids,attention_mask=attention_mask)
        
        features = x[0]
        features = features[:,0,:]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)
            # features = self.relu(features)

        return features