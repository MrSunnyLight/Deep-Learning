import os
import random
from scipy import io as scio
import argparse


def convert_CUB200(data_root):
    images_txt = os.path.join(data_root, 'images.txt')
    train_val_txt = os.path.join(data_root, 'train_test_split.txt')
    labels_txt = os.path.join(data_root, 'image_class_labels.txt')

    id_name_dict = {}
    id_class_dict = {}
    id_train_val = {}
    with open(images_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, name = line.strip().split()
            id_name_dict[id] = name
            line = f.readline()

    with open(train_val_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, trainval = line.strip().split()
            id_train_val[id] = trainval
            line = f.readline()

    with open(labels_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, class_id = line.strip().split()
            id_class_dict[id] = int(class_id)
            line = f.readline()

    train_txt = os.path.join(data_root, 'CUB200_train.txt')
    test_txt = os.path.join(data_root, 'CUB200_test.txt')
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(test_txt):
        os.remove(test_txt)

    f1 = open(train_txt, 'a', encoding='utf-8')
    f2 = open(test_txt, 'a', encoding='utf-8')

    for id, trainval in id_train_val.items():
        if trainval == '1':
            f1.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
        else:
            f2.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
    f1.close()
    f2.close()


if __name__ == '__main__':
    # convert_CUB200('../../datasets/CUB_200_2011')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='CUB200')
    parser.add_argument('--root_path', type=str, default='../../datasets/CUB_200_2011')
    arg = parser.parse_args()
    func = eval('convert_' + arg.dataset_name)      # 找到并返回这个函数对象
    func(arg.root_path)     # 调用获取到的函数对象
