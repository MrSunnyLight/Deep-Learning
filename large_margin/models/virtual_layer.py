import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VirtualSoftmax(nn.Module):
    def __init__(self, in_features, num_classes, init_weights=False, scale=1):
        super(VirtualSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        if init_weights:
            nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels, mode):
        weight = self.weight.T  # weight(512,10)
        # weight = F.normalize(weight, dim=0)     #******* fc层权重进行归一化
        WX = torch.matmul(inputs, weight)  # WX(128,10)
        if mode:
            W_yi = weight[:, labels]  # W_yi(512,128)
            W_yi_norm = torch.norm(W_yi, dim=0)  # W_yi_norm(128,)
            X_i_norm = torch.norm(inputs, dim=1)  # X_i_norm(128,)
            WX_virt = W_yi_norm * X_i_norm * self.scale  # WX_virt(128,)
            WX_virt = torch.clamp(WX_virt, min=1e-10, max=15.0)
            WX_virt = WX_virt.unsqueeze(1)  # WX_virt(128,1)
            WX_new = torch.cat([WX, WX_virt], dim=1)  # WX_new(128,11)
            return WX_new
        else:
            return WX


class MSCEVirtualSoftmax(nn.Module):
    def __init__(self, prototypes):
        super(MSCEVirtualSoftmax, self).__init__()

    def forward(self):
        pass


class VirtualSoftmaxLearningStrategy(nn.Module):
    def __init__(self, in_features, num_classes, init_weights=False, scale=1):
        super(VirtualSoftmaxLearningStrategy, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        if init_weights:
            nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels, mode, epoch=None):
        weight = self.weight.T
        WX = torch.matmul(inputs, weight)
        if mode and epoch is None:
            # get label indices to get W_yi
            W_yi = weight[:, labels].t()  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=1)
            X_i_norm = torch.norm(inputs, dim=1)
            # 计算输入张量和权重张量的夹角
            cosine_sim = torch.nn.functional.cosine_similarity(inputs, W_yi, dim=1)
            angle = torch.acos(cosine_sim) * (180.0 / 3.14159)  # 弧度转换为角度
            max_angle = max(angle)
            mask_simple = angle < max_angle / 3
            # mask_hopeful = (max_angle / 3) < angle < (max_angle / 3 * 2)
            mask_hopeful = torch.gt(angle, max_angle / 3) & torch.lt(angle, (max_angle / 3 * 2))
            mask_difficult = angle > max_angle / 3 * 2
            # 简单样本，用静态负类
            sip_W_yi = torch.negative(W_yi) * self.scale
            sip_WX_virt = torch.einsum('ij,ij->i', sip_W_yi, inputs)
            sip_WX_virt[~mask_simple] = 0
            # 有希望的样本，用动态合成类
            hop_W_virt = F.normalize(inputs)
            # 合成的虚拟负类，S=z_i-W_yi, w_virt=|W|(S/|S|)：
            hop_W_virt = F.normalize(hop_W_virt - W_yi) * W_yi_norm.unsqueeze(1) * self.scale
            hop_WX_virt = torch.einsum('ij,ij->i', hop_W_virt, inputs)
            hop_WX_virt[~mask_hopeful] = 0
            # 困难样本，用动态虚拟类
            dif_WX_virt = W_yi_norm * X_i_norm * self.scale
            dif_WX_virt[~mask_difficult] = 0

            WX_virt = sip_WX_virt + hop_WX_virt + dif_WX_virt
            WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class MSCEVirtualSoftmaxLearningStrategy(nn.Module):
    def __init__(self, prototypes, scale=1):
        super(MSCEVirtualSoftmaxLearningStrategy, self).__init__()
        self.weight = nn.Parameter(prototypes, requires_grad=False)  # weight:(9,10)
        self.scale = scale

    def forward(self, inputs, labels, mode, dataset=None):
        # calculate normal WX(output of final FC)
        WX = torch.matmul(inputs, self.weight)
        strategy = 1
        if mode:
            # get label indices to get W_yi
            W_yi = self.weight[:, labels].t()  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=1)
            X_i_norm = torch.norm(inputs, dim=1)
            # 计算输入张量和权重张量的夹角
            cosine_sim = torch.nn.functional.cosine_similarity(inputs, W_yi, dim=1)
            angle = torch.acos(cosine_sim) * (180.0 / 3.14159)  # 弧度转换为角度
            angle2 = [88, 100, 90]
            max_angle = angle2[dataset]
            # max_angle = max(angle)
            mask_simple = angle < max_angle / 3
            # mask_hopeful = (max_angle / 3) < angle < (max_angle / 3 * 2)
            mask_hopeful = torch.gt(angle, max_angle / 3) & torch.lt(angle, (max_angle / 3 * 2))
            mask_difficult = angle > max_angle / 3 * 2
            # 简单样本，用静态负类
            sip_W_yi = torch.negative(W_yi) * self.scale
            sip_WX_virt = torch.einsum('ij,ij->i', sip_W_yi, inputs)
            sip_WX_virt[~mask_simple] = 0
            # 有希望的样本，用动态合成类
            hop_W_virt = F.normalize(inputs)
            # 合成的虚拟负类，S=z_i-W_yi, w_virt=|W|(S/|S|)
            hop_W_virt = F.normalize(hop_W_virt - W_yi) * W_yi_norm.unsqueeze(1) * self.scale
            hop_WX_virt = torch.einsum('ij,ij->i', hop_W_virt, inputs)
            hop_WX_virt[~mask_hopeful] = 0
            # 困难样本，用动态虚拟类
            dif_WX_virt = W_yi_norm * X_i_norm * self.scale
            dif_WX_virt[~mask_difficult] = 0

            WX_virt = sip_WX_virt + hop_WX_virt + dif_WX_virt
            WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class ResultantVirtualSoftmax(nn.Module):  # 合成的虚拟负类，w_virt=scele * (z_i-W_yi)/|(z_i-W_yi)|*|W_yi|
    def __init__(self, in_features, num_classes, init_weights=False, scale=1):
        super(ResultantVirtualSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        if init_weights:
            nn.init.xavier_uniform_(self.weight.data)

    def forward_backup(self, inputs, labels, mode):
        weight = self.weight.T
        WX = torch.matmul(inputs, weight)
        if mode:
            W_yi = weight[:, labels]  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=0).unsqueeze(1)
            W_virt = inputs - W_yi.t()
            W_virt_norm = torch.norm(W_virt, dim=1).unsqueeze(1)

            W_virt = W_virt / W_virt_norm * W_yi_norm * self.scale  # 合成的虚拟负类，w_virt=scele * (z_i-W_yi)/|(z_i-W_yi)|*|W_yi|
            WX_virt = torch.einsum('ij,ij->i', W_virt, inputs)
            WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX

    def forward(self, inputs, labels, mode):
        weight = self.weight.T
        WX = torch.matmul(inputs, weight)
        if mode:
            W_yi = weight[:, labels]  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=0).unsqueeze(1)
            W_virt = inputs - W_yi.t()
            W_virt = F.normalize(W_virt, dim=1) * W_yi_norm * self.scale

            WX_virt = torch.einsum('ij,ij->i', W_virt, inputs)
            WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class MSCEResultantVirtualSoftmax(nn.Module):  # 合成的虚拟负类，w_virt=1/2(z_i-W_yi)
    def __init__(self, prototypes, scale=0.5):
        super(MSCEResultantVirtualSoftmax, self).__init__()
        self.weight = nn.Parameter(prototypes, requires_grad=False)  # weight:(9,10)
        self.scale = scale

    def forward(self, inputs, labels, mode):
        WX = torch.matmul(inputs, self.weight)
        if mode:
            W_yi = self.weight[:, labels]  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=0)
            X_i_norm = torch.norm(inputs, dim=1)

            # W_virt = inputs / X_i_norm.unsqueeze(1) * W_yi_norm.unsqueeze(1)
            W_virt = F.normalize(inputs)
            W_virt = (W_virt - W_yi.t()) * self.scale  # 合成的虚拟负类，w_virt=1/2(z_i-W_yi)
            # modified on 4-20-14 by huang
            # W_virt = W_virt* self.scale - W_yi.t() * (1-self.scale ) # 合成的虚拟负类，w_virt=m*z_i-(1-m)*W_yi

            WX_virt = torch.einsum('ij,ij->i', W_virt, inputs)
            WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)  # for numerical stability
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class LargestVirtual(nn.Module):
    # 原始virtual： v = (|W_yi| / |z_i|) * z_i
    # 合成的： s = (|W_yi| / |h|) * h, where h = (z_i - W_yi)/(|z_i - W_yi|)
    def __init__(self, in_features, num_classes, init_weights=False, scale=1):
        super(LargestVirtual, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        if init_weights:
            nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels, mode, select=None):
        weight = self.weight.T
        WZ = torch.matmul(inputs, weight)
        if mode:
            W_yi = weight[:, labels]  # 选择与每个输入样本类别相对应的权重 (9,128)
            W_yi_norm = torch.norm(W_yi, dim=0)
            z_i_norm = torch.norm(inputs, dim=1)

            v_zi = W_yi_norm * z_i_norm
            v_zi = torch.clamp(v_zi, min=1e-10, max=15.0)
            v_zi = v_zi.unsqueeze(1)

            # modified on 4-20-14 by huang
            # h = F.normalize(inputs, dim=1) - F.normalize(W_yi.t(), dim=1)
            z_i_unit=F.normalize(inputs, dim=1)
            w_yi_unit=F.normalize(W_yi.t(), dim=1)
            h = z_i_unit -  w_yi_unit

            s = W_yi_norm.unsqueeze(1) * F.normalize(h, dim=1) * self.scale
            s_zi = torch.einsum('ij,ij->i', s, inputs)
            s_zi = torch.clamp(s_zi, 1e-10, 15.0)
            s_zi = s_zi.unsqueeze(1)
            s_zi_2 = s_zi * 2

            _, max_indices = WZ.max(dim=1)
            if select == 1:     # 仅用v
                selected = v_zi
            elif select == 2:   # 仅用s
                selected = s_zi
            elif select == 3:   # 仅用2s
                selected = s_zi_2
            elif select == 4:   # 当前是最大则用s，否则用2s
                selected = torch.where(max_indices.unsqueeze(1) == labels.unsqueeze(1), s_zi, s_zi_2)
            elif select == 5:   # 当前是最大则用s，否则用v
                # 创建一个选择向量，如果行的最大值索引与label相同，则选择s，否则选择v
                selected = torch.where(max_indices.unsqueeze(1) == labels.unsqueeze(1), s_zi, v_zi)
            elif select == 6:   # 当前是最大则用2s，否则用v
                selected = torch.where(max_indices.unsqueeze(1) == labels.unsqueeze(1), s_zi_2, v_zi)
            elif select == 7:   # 仅用 2 * -W_yi
                nvc = W_yi.t() * 2     # (negative virtual class, nvc)
                selected = torch.einsum('ij,ij->i', nvc, inputs).unsqueeze(1)
            elif select == 8:
                h = self.scale * z_i_unit - (1 - self.scale) * w_yi_unit  # w_yi: column vectors; w_yi_unit and inputs: row vectors
                h = W_yi_norm.unsqueeze(1) * F.normalize(h, dim=1) # h is a row vectors, now the length=w_yi
                selected = torch.einsum('ij,ij->i', h, inputs).unsqueeze(1)
                # print('enter select==8, the scale={:.2f} '.format(self.scale))

            else:
                raise Exception("select: {} does not meet any of the required conditions".format(select))
            WZ_new = torch.cat((WZ, selected), dim=1)
            return WZ_new
        else:
            return WZ
