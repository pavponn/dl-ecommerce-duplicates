from modules.datasets.ArcfaceLossShopeeDataset import ArcfaceLossShopeeDataset
from PIL import Image

class ImageArcFaceLossShopeeDataset(ArcFaceLossShopeeDataset):
	def __init__(self, x, y, transform=None):
		super().__init__(x, y, transform)

	def _transform_to_actual_dataset_item(self, item):
		return Image.open(item).convert('RGB')