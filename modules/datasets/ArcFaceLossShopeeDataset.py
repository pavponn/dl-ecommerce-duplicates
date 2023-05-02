import torch
from torch.utils.data import Dataset
import abc

class ArcFaceLossShopeeDataset(Dataset, abc.ABC):
	def __init__(self, x, y, transform):
		assert len(x) == len(y)
		self.x = x
		self.y = y
		self.transform = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		x = self.x[index]
		y = self.y[index]

		x = self._transform_to_actual_dataset_item(x)

		x = self.transform(x)

		return x, torch.tensor(y, dtype=torch.float32)

	@abc.abstractmethod
	def _transform_to_actual_dataset_item(self, item):
		"""
		Takes an element of a dataset and transforms it into actual value
		(e.g., if the item is an image path, then this method will open image by this path)
		"""
		pass