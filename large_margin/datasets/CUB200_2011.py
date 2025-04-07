import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, images_path_list, images_label_list, image_root, transform=None):
        self.images_path_list = images_path_list
        self.images_label_list = images_label_list
        self.image_root = image_root
        self.num_classes = max(self.images_label_list) + 1
        self.transform = transform

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, item):
        img_name = self.images_path_list[item]
        img_name = os.path.join(self.image_root, img_name)
        img = Image.open(img_name)
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            # print("image: {} isn't RGB mode.".format(self.images_path_list[item]))
            img = img.convert('RGB')
        label = self.images_label_list[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


