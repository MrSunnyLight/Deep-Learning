import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinear(nn.Module):  # 对权重和输入进行L2正则化
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight.data)

    def forward(self, input):
        input = F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)


class NormLinear_for_msce_LMSoftmax(nn.Module):  # 对权重和输入进行L2正则化
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, prototypes):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(prototypes.t(), requires_grad=False)

    def forward(self, input):
        input = F.normalize(input, dim=1)
        weight = self.weight
        # weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)
