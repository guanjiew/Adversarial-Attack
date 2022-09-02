import torch.utils.data as data
from PIL import Image

class CustomDataset(data.Dataset):

    def __init__(self, X, y, transform=None):

        self.features = X
        self.label = y
        self.transfrom = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        X = Image.fromarray(X, 'L')
        if self.transfrom:
            X = self.transfrom(X)
        y = self.label[index]
        return X, y