import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .transforms import get_transform

NUM_WORKER = os.cpu_count()

def create_dataloader(data_dir: str, batch_size: int, image_size: int, num_worker: int):
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')
    
    train_transforms = get_transform(image_size=image_size, is_train=True)
    test_transforms = get_transform(image_size=image_size, is_train=False)

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return train_loader, test_loader, class_names