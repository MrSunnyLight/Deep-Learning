import argparse

import numpy as np
import torch

import create_max_separated_matrix


def load_prototypes(args, device):
    case = args.load_p
    num_classes = 100 if args.dataset == "cifar100" else 10
    prototypes = torch.from_numpy(create_max_separated_matrix.create_prototypes(num_classes)).float()  # (n, n-1)
    prototypes *= args.radius
    dims = prototypes.shape[1]
    prototypes = prototypes.t().to(device)  # (n-1, n)
    test_prototypes = prototypes

    if case == 1:       # 默认情况，test_prototypes 与train相同，来自 args
        pass
    elif case == 2:     # test_prototypes 使用基础的10个，即 prototypes10.npy
        test_prototypes = torch.from_numpy(np.load('prototypes10.npy')).float()
        test_prototypes *= args.radius
        test_prototypes = test_prototypes.t().to(device)
    elif case == 2.5:   # virtul prototypes乘以一个倍率缩放
        s = args.s           # 特征长度缩放系数
        # s = 0.1
        # s = 0.2
        prototypes[:, 10:] *= s
        print("rotated_vectors_norm", torch.norm(prototypes, dim=0))
        test_prototypes = torch.from_numpy(np.load('prototypes10.npy')).float()
        test_prototypes *= args.radius
        test_prototypes = test_prototypes.t().to(device)
    elif case == 3:     # prototypes 和 test_prototypes 都调换第i列和第j列
        i = args.exchange
        j = 9
        prototypes[:, [i, j]] = prototypes[:, [j, i]]
        test_prototypes[:, [i, j]] = test_prototypes[:, [j, i]]
    elif case == 4:     # 12 维，从9维10个基础上补0扩展到12维10个
        expand = torch.zeros(3, 10).to(device)
        # print(prototypes)
        # print(prototypes.shape)
        prototypes = torch.cat((expand, prototypes), dim=0)
        # print(prototypes)
        # print(prototypes.shape)
        # print(test_prototypes)
        # print(test_prototypes.shape)
        test_prototypes = prototypes
        dims = prototypes.shape[0]
    elif case == 5:     # 12维，16个。从“4”的基础上，右边添加6个向量
        expand_top = torch.zeros(3, 10).to(device)
        expand_right = torch.zeros(12, 6).to(device)
        expand_right[0, 0] = 1
        expand_right[0, 1] = -1
        expand_right[1, 2] = 1
        expand_right[1, 3] = -1
        expand_right[2, 4] = 1
        expand_right[2, 5] = -1
        # print(expand_top)
        # print(expand_right)
        prototypes = torch.cat((expand_top, prototypes), dim=0)
        test_prototypes = prototypes
        prototypes = torch.cat((prototypes, expand_right), dim=1)
        # print(prototypes)
        # print(prototypes.shape)
        dims = prototypes.shape[0]
    elif case == 5.5:       # 12维，加6个垂直和10个反向，共26个
        expand_top = torch.zeros(3, 20).to(device)
        expand_right = torch.zeros(12, 6).to(device)
        for i in range(0, 6, 2):
            expand_right[i//2, i] = 1
            expand_right[i//2, i+1] = -1
        prototypes_minus = torch.negative(prototypes)
        prototypes = torch.cat((prototypes, prototypes_minus), dim=1)
        prototypes = torch.cat((expand_top, prototypes), dim=0)
        prototypes = torch.cat((prototypes, expand_right), dim=1)
        expand_test_top = torch.zeros(3, 10).to(device)
        test_prototypes = torch.cat((expand_test_top, test_prototypes), dim=0)
        dims = prototypes.shape[0]
    elif case == 6:     # 15维，添加12个，变成22个向量，测试时变为0
        expand_top = torch.zeros(6, 10).to(device)
        expand_right = torch.zeros(15, 12).to(device)
        for i in range(0, 12, 2):
            expand_right[i//2, i] = 1
            expand_right[i//2, i+1] = -1
        # print(expand_right)
        prototypes = torch.cat((expand_top, prototypes), dim=0)
        test_prototypes = prototypes
        prototypes = torch.cat((prototypes, expand_right), dim=1)
        dims = prototypes.shape[0]
    elif case == 7:     # 既取反向，又添加12个，共10+10+12=32个向量，15维
        expand_top = torch.zeros(6, 20).to(device)
        expand_right = torch.zeros(15, 12).to(device)
        for i in range(0, 12, 2):
            expand_right[i//2, i] = 1
            expand_right[i//2, i+1] = -1
        prototypes_minus = torch.negative(prototypes)
        prototypes = torch.cat((prototypes, prototypes_minus), dim=1)
        prototypes = torch.cat((expand_top, prototypes), dim=0)
        prototypes = torch.cat((prototypes, expand_right), dim=1)
        test_prototypes = torch.from_numpy(np.load('prototypes10.npy')).float()
        test_prototypes *= args.radius
        test_prototypes = test_prototypes.t().to(device)
        expand_test_top = torch.zeros(6, 10).to(device)
        test_prototypes = torch.cat((expand_test_top, test_prototypes), dim=0)
        dims = prototypes.shape[0]
    elif case == 8:     # test_prototypes只保留前10个向量
        test_prototypes = test_prototypes[:, :10]
    elif case == 9:     # train的时候增加一倍的虚拟反向，test的时候还是原来的
        prototypes_minus = torch.negative(prototypes)
        prototypes = torch.cat((prototypes, prototypes_minus), dim=1)
        print(prototypes.shape)
        # test_prototypes = test_prototypes[:, :10]
        print(test_prototypes.shape)
    else:
        pass
    print("prototypes shape:", prototypes.shape)
    print("dims:", dims)
    return prototypes, test_prototypes, dims


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototypes', type=str, default="prototypes10.npy", help='分离矩阵的路径')
    parser.add_argument("--radius", default=1.0, type=float, help="prototypes半径")
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--exchange', default=0, type=int, help='交换号')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prototypes, test_prototypes, dims = load_prototypes(args, device)
