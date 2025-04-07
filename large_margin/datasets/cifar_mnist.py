import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision import transforms as transforms
from datasets.CUB200_2011 import CustomDataset


def get_cifar100(data_root="../../datasets/", batch_size=128, num_workers=0):
    # Mean and std pixel values.
    cmean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cstd = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Transforms.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    train_dataset = CIFAR100(root=data_root, train=True, transform=transform_train, download=True)
    test_dataset = CIFAR100(root=data_root, train=False, transform=transform_test, download=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, test_dataloader

# modified by huang 16-6-24  data_root
def get_cifar10(data_root="./datasets/", batch_size=128, num_workers=0):
    # Mean and std pixel values.
    cmean = (0.4914, 0.4822, 0.4465)
    cstd = (0.2023, 0.1994, 0.2010)
    # Transforms.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    train_dataset = CIFAR10(root=data_root, train=True, transform=transform_train, download=True)
    test_dataset = CIFAR10(root=data_root, train=False, transform=transform_test, download=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_mnist(data_root="../../datasets/", batch_size=128, num_workers=0):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
    train_dataset = MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_root, train=False, transform=transform, download=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_SVHN(data_root="../../datasets/", batch_size=128, num_workers=0):
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机颜色抖动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.106], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.106], std=[0.229, 0.224, 0.225])
        ])
    }
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.106], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    train_dataset = SVHN(root=data_root, split='train', download=True, transform=transform)
    test_dataset = SVHN(root=data_root, split='test', download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_CUB200(data_root="../../datasets/", batch_size=128, num_workers=0):
    image_root = os.path.join(data_root, 'images')
    assert os.path.exists(image_root), "dataset root: {} does not exist.".format(image_root)
    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []
    train_category_counts = {}
    test_category_counts = {}

    train_txt = os.path.join(data_root, 'CUB200_train.txt')
    test_txt = os.path.join(data_root, 'CUB200_test.txt')
    with open(train_txt, 'r') as f:
        line = f.readline()
        while line:
            img_name = line.split()[0]
            label = int(line.split()[1])
            train_images_path.append(img_name)
            train_images_label.append(label)
            train_category_counts[label] = train_category_counts.get(label, 0) + 1
            line = f.readline()
    with open(test_txt, 'r') as f:
        line = f.readline()
        while line:
            img_name = line.split()[0]
            label = int(line.split()[1])
            test_images_path.append(img_name)
            test_images_label.append(label)
            test_category_counts[label] = test_category_counts.get(label, 0) + 1
            line = f.readline()
    print("{} images were found in the dataset.".format(len(train_images_path) + len(test_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    print("Number of samples for each class in training set is: {}".format(train_category_counts))
    print("{} images for testing.".format(len(test_images_path)))
    print("Number of samples for each class in test set is: {}".format(test_category_counts))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(512),  # 256
                                     transforms.RandomCrop(448),  # 224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(512),
                                    transforms.CenterCrop(448),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    train_dataset = CustomDataset(train_images_path, train_images_label, image_root, data_transform["train"])
    test_dataset = CustomDataset(test_images_path, test_images_label, image_root, data_transform["test"])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader
