import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageArcFaceLossShopeeDataset(Dataset):
	def __init__(self, df, mode, transform=None):
		self.df = df.reset_index(drop=True)
		self.mode = mode
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		row = self.df.loc[index]
		img = cv2.imread(row.file_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if self.transform is not None:
			tf = self.transform(image=img)
			img = tf['image']
		img = img.astype(np.float32)
		img = img.transpose(2, 0, 1)
		if self.mode == 'test':
			return torch.tensor(img).float()
		else:
			return torch.tensor(img).float(), torch.tensor(row.label_group).float()