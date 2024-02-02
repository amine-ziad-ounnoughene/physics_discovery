import torch
from torch.utils.data import Dataset, DataLoader

class SciNet_loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'vars': torch.tensor(self.data[idx]['vars'], dtype=torch.float32),
            'timeserie': torch.tensor(self.data[idx]['timeserie'], dtype=torch.float32),
            'question': torch.tensor(self.data[idx]['question'], dtype=torch.float32),
            'answer': torch.tensor(self.data[idx]['answer'], dtype=torch.float32)
        }
        return sample