import abc

import torch
from torch.utils.data import Dataset


class TripletLossShopeeDataset(Dataset, abc.ABC):
    """
    Abstract class for all datasets that are used
    to train network using contrastive loss (i.e., have 2 objects with a label "are same").
    """

    def __init__(self, xs_1, xs_2, xs_3, ys, transform):
        assert len(xs_1) == len(xs_2) and len(xs_1) == len(ys)
        self.xs_1 = xs_1
        self.xs_2 = xs_2
        self.xs_3 = xs_3
        self.ys = ys
        self.transform = transform

    def __getitem__(self, index):
        x_1 = self.xs_1[index]
        x_2 = self.xs_2[index]
        x_3 = self.xs_2[index]
        y = self.ys[index]

        x_1 = self._transform_to_actual_dataset_item(x_1)
        x_2 = self._transform_to_actual_dataset_item(x_2)
        x_3 = self._transform_to_actual_dataset_item(x_3)

        x_1 = self.transform(x_1)
        x_2 = self.transform(x_2)
        x_3 = self.transform(x_3)

        return x_1, x_2, x_3,torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.xs_1)

    @abc.abstractmethod
    def _transform_to_actual_dataset_item(self, item):
        """
        Takes an element of a dataset and transforms it into actual value
        (e.g., if the item is an image path, then this method will open image by this path)
        """
        pass
