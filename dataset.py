import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     return self.data[idx], self.labels[idx]
    
    def __getitem__(self, idx):
        # Convert to tensor 
        embedding_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)  
        return embedding_tensor, label_tensor