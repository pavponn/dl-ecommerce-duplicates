from torch.utils.data import Dataset
from PIL import Image


class ImageWithLabelsShopeeDataset(Dataset):
    def __init__(self, img_path, label, transform):
        self.img_path = img_path
        self.label = label
        assert len(self.img_path) == len(self.label)
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img, self.label[index]

