from datasets.cifar_mnist import *

# modified by huang 16-6-24  data_root
def load_dataset(dataset_name, data_root='./datasets/', batch_size=128, num_workers=0):
    name = ('mnist', 'cifar10', 'cifar100', 'SVHN', 'CUB200-2011')
    assert dataset_name in name
    print("loading normal {} dataset ...".format(dataset_name))
    if dataset_name == 'cifar10':
        return get_cifar10(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    if dataset_name == 'cifar100':
        return get_cifar100(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    if dataset_name == 'mnist':
        return get_mnist(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    if dataset_name == 'SVHN':
        return get_SVHN(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    if dataset_name == 'CUB200-2011':
        return get_CUB200(data_root=data_root, batch_size=batch_size, num_workers=num_workers)



if __name__ == '__main__':
    train_loader, test_loader = load_dataset(dataset_name="CUB200-2011", data_root='../../datasets/CUB_200_2011', batch_size=128,
                                             num_workers=8)
    print("len(train_loader)", len(train_loader))
    print("len(train_loader.dataset)", len(train_loader.dataset))

    len_images = []
    for i, (images, labels) in enumerate(train_loader):
        len_images.append(len(images))
        if i == 0:
            print(labels)
    print("len_images", len_images)

    print("len(test_loader)", len(test_loader))
    print("len(test_loader.dataset)", len(test_loader.dataset))
    len_images.clear()
    for (images, labels) in test_loader:
        len_images.append(len(images))
    print("len_images", len_images)
