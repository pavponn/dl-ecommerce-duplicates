from tqdm import tqdm
import torch
import numpy as np


class EmbeddingsProducer(object):
    """
    Class to produce embeddings on various data loaders given a model and device.
    """

    def __init__(self, model: torch.nn.Module, device):
        self.model = model
        self.device = device

    def get_embeddings(self, loader, normalize=False):
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for data in tqdm(loader):
                data = data.to(self.device)
                embeddings = self.model(data)
                embeddings = embeddings.data.cpu().numpy()
                all_embeddings.extend(embeddings)
        all_embeddings = np.stack(all_embeddings)
        if normalize:
            norms = np.linalg.norm(all_embeddings, axis=1)
            all_embeddings = all_embeddings / norms[:, np.newaxis]
        return all_embeddings
