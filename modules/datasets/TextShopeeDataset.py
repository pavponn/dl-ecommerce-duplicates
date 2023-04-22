from torch.utils.data import Dataset


class TextShopeeDataset(Dataset):
    def __init__(self, text):
        self.text = text
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]
