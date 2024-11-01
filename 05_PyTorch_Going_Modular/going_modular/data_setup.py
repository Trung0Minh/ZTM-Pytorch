
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False,
                                 pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
