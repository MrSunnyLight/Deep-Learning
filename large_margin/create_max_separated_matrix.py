import sys
import torch
import numpy as np
from scipy.spatial.distance import *

# 生成具有最大分离度的矩阵的代码。
# 您还可以通过调用create_prototypes()函数直接在代码中加载它，而无需将其保存为npy文件。

sys.setrecursionlimit(10000)  # for nr_prototypes>=1000   设置python对递归函数深度的默认值


def create_noisy_prototypes(nr_prototypes, noise_scale=0):  # 生成带有高斯噪声的原型矩阵
    prototypes = create_prototypes(nr_prototypes)
    if noise_scale != 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=prototypes.shape)
        prototypes = L2_norm(prototypes + noise)
    distances = cdist(prototypes, prototypes)
    avg_dist = distances[~np.eye(*distances.shape, dtype=bool)].mean()
    return prototypes.astype(np.float32), avg_dist


def create_prototypes(nr_prototypes):
    assert nr_prototypes > 0
    prototypes = V(nr_prototypes - 1).T
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    distances = cdist(prototypes, prototypes)
    assert distances[~np.eye(*distances.shape, dtype=bool)].std() <= 1e-3
    return prototypes.astype(np.float32)


def create_prototypes_random(nr_prototypes):  # 生成随机的原型矩阵
    prototypes = L2_norm(np.random.uniform(size=(nr_prototypes, nr_prototypes - 1)))
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    return prototypes.astype(np.float32)


def V(order):  # 创建具有特定顺序的Vandermonde矩阵
    if order == 1:
        return np.array([[1, -1]])
    else:
        col1 = np.zeros((order, 1))  # 创建order行1列元素为0的2维矩阵
        col1[0] = 1
        row1 = -1 / order * np.ones((1, order))  # 创建一个1行order列，每个元素都是 -1/order 的 np 数组
        return np.concatenate((col1, np.concatenate((row1, np.sqrt(1 - 1 / (order ** 2)) * V(order - 1)), axis=0)),
                              axis=1)


if __name__ == '__main__':  # 当该文件作为脚本时，运行该函数来生成原型矩阵并将其保存为.npy文件。
    nr_classes = 129
    prototypes = create_prototypes(nr_classes)
    np.save("prototypes"+str(nr_classes)+".npy", prototypes)
    print(prototypes)

    # # 分离矩阵增加1维，变成10维11个向量
    # padded_prototypes = np.pad(prototypes, ((1, 0), (1, 0)), mode='constant')
    # padded_prototypes[0][0] = 1
    # print(padded_prototypes)
    # print(padded_prototypes.shape)
    # np.save("prototypes-11"+".npy", padded_prototypes)
    #
    # # 维度不变，变成10维12个向量
    # col = np.zeros((1, 10))
    # col[0][0] = -1
    # padded_prototypes2 = np.concatenate((padded_prototypes, col), axis=0)
    # print(padded_prototypes2)
    # print(padded_prototypes2.shape)
    # np.save("prototypes-12"+".npy", padded_prototypes2)
    #
    # # stack到一起，变成9维20个向量
    # prototypes = prototypes.T
    # prototypes_minus = np.negative(prototypes)
    # print(prototypes_minus)
    # print(prototypes_minus.shape)
    # concat = np.concatenate((prototypes, prototypes_minus), axis=1)
    # print(concat)
    # print(concat.shape)
    # concat = concat.T
    # print(concat.shape)
    # np.save("prototypes-20" + ".npy", concat)

    # prototypes = torch.from_numpy(prototypes).float()
    # prototypes = prototypes.t()
    # print(prototypes)



