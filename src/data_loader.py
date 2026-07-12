import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size, valid_split, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    num_train = int(len(train_dataset) * (1 - valid_split))
    num_valid = len(train_dataset) - num_train
    train_subset, valid_subset = random_split(
        train_dataset, [num_train, num_valid],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader