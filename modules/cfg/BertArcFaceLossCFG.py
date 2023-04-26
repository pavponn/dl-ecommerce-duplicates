import torch

class BertArcFaceLossCFG:
    DistilBERT = True # if set to False, BERT model will be used
    bert_hidden_size = 768
    batch_size = 64
    epochs = 30
    num_workers = 4
    learning_rate = 1e-5 #3e-5
    scheduler = "ReduceLROnPlateau"
    step = 'epoch'
    patience = 2
    factor = 0.8
    dropout = 0.5
    model_path = "/content/gdrive/MyDrive/dl-ecommerce-duplicates/checkpoints"
    max_length = 30
    model_save_name = "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')