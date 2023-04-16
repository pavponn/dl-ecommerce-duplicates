from modules.datasets.ContrastiveLossShopeeDataset import ContrastiveLossShopeeDataset
from PIL import Image


class ImageContrastiveLossShopeeDataset(ContrastiveLossShopeeDataset):
    def __init__(self, xs_1, xs_2, ys, transform):
        super().__init__(xs_1, xs_2, ys, transform)

    def _transform_to_actual_dataset_item(self, item):
        return Image.open(item).convert('RGB')
