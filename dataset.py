import torch
from torch.utils.data import Dataset
from PIL import Image


class SingleStyleData(Dataset):
    def __init__(self, path_content, path_style, device, transform=None):
        super(SingleStyleData, self).__init__()
        self.path_content = path_content
        self.path_style = path_style
        self.transform = transform
        self.device = device
        self.content, self.style = self._get_data()

    def _get_data(self):
        content = Image.open(self.path_content).convert('RGB')
        style = Image.open(self.path_style).convert('RGB')

        content = self.transform(content).unsqueeze(0)
        style = self.transform(style).unsqueeze(0)

        return content.to(self.device, torch.float), style.to(self.device, torch.float)

    def __getitem__(self, index):
        return self.content, self.style

    def __len__(self):
        return 1
