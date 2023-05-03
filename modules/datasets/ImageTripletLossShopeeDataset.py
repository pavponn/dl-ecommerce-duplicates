from modules.datasets.TripletLossShopeeDataset import TripletLossShopeeDataset
from PIL import Image


class ImageTripletLossShopeeDataset(TripletLossShopeeDataset):
    def __init__(self, xs_1, xs_2,xs_3, ys, transform):
        super().__init__(xs_1, xs_2, xs_3,ys, transform)

    def _transform_to_actual_dataset_item(self, item):
        return Image.open(item).convert('RGB')
