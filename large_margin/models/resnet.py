"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import torchinfo

import create_max_separated_matrix
from models.norm_layer import NormLinear, NormLinear_for_msce_LMSoftmax
from models.virtual_layer import *

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34 """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        # return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        # return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, prototypes=None, num_classes=100, norm=False, scale=None, init_weights=False,
                 mode=''):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True))
            nn.LeakyReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if mode == 'msce_virtual':  # 加固定的prototypes
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.classifier = MSCEVirtualSoftmax(prototypes)
        elif mode == 'msce_virtual_learning_strategy':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.classifier = MSCEVirtualSoftmaxLearningStrategy(prototypes)
        elif mode == 'msce_resultant_virtual':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.classifier = MSCEResultantVirtualSoftmax(prototypes)
        elif mode == 'msce_LMSoftmax':  # 用LMSoftmax，feature必须归一化，否则直接训练loss直接崩
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.classifier = NormLinear_for_msce_LMSoftmax(num_classes, num_classes + 1, prototypes)
        elif 'msce' in mode:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.classifier = nn.Linear(num_classes, num_classes + 1, bias=False)
        else:  # 不多加固定的prototypes
            if mode in ['virtual_softmax', 'virtual_softmax_rsm', 'virtual_focal']:
                self.fc = VirtualSoftmax(512 * block.expansion, num_classes, init_weights=init_weights, scale=scale)
            elif mode in ['resultant_virtual']:
                self.fc = ResultantVirtualSoftmax(512 * block.expansion, num_classes, init_weights=init_weights,
                                                  scale=scale)
            elif mode in ['virtual_learning_strategy']:
                self.fc = VirtualSoftmaxLearningStrategy(512 * block.expansion, num_classes, init_weights=init_weights,
                                                         scale=scale)
            elif mode in ['virtual_learning_strategy_addfc']:
                self.add_fc = nn.Linear(512 * block.expansion, num_classes - 1)
                self.fc = VirtualSoftmaxLearningStrategy(num_classes - 1, num_classes, init_weights=init_weights,
                                                         scale=scale)
            elif mode in ['largest_virtual']:
                self.fc = LargestVirtual(512 * block.expansion, num_classes, init_weights=init_weights, scale=scale)
            else:  # 'toward_largest_margin' or 'sce'
                self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)
                # self.fc = NormLinear(512 * block.expansion, num_classes)  # 用sce，loss为LMSoftmax时使用

        self.norm = norm
        self.scale = scale      # 对feature的缩放
        self.mode = mode
        if init_weights:  # 是否初始化权重
            self._initialize_weights()
        if mode == 'msce':
            with torch.no_grad():
                self.classifier.weight.data = nn.Parameter(prototypes.t())
            for param in self.classifier.parameters():
                param.requires_grad = False

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, target=None, s=1):
        output = self.get_body(x)
        output = self.fc(output)  # 复现virtual class论文的时候，要注释掉
        if self.norm:
            output = F.normalize(output, p=2, dim=1)
        if self.scale is not None:
            output = output * self.scale
        if self.training and target is not None:  # 在训练模式下，添加虚拟负类
            output = self.classifier(output, target, s)
        return output

    def get_body(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        if 'addfc' in self.mode:
            output = self.add_fc(output)
        if 'msce' in self.mode:
            output = self.fc(output)
        return output

    def get_weight(self):  # Get the weight of the last fully-connected layer 返回最后一个全连接层的权重
        if 'msce' in self.mode:
            if 'virtual' in self.mode:
                return self.classifier.weight.T
            else:
                return self.classifier.weight
        else:
            return self.fc.weight

    def _initialize_weights(self):  # 重置网络参数，用 kaiming_uniform_或xavier_uniform_ 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 若为卷积层，以ReLU的非线性方式对卷积层的权重进行均匀分布初始化
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 若为线性层，则对全连接层的权重进行均匀分布初始化
        return


def resnet18(dims, prototypes, norm=False, init_weights=False, scale=None, mode='sce'):
    """ return a ResNet 18 object """
    return ResNet(BasicBlock, [2, 2, 2, 2], prototypes, num_classes=dims, norm=norm, init_weights=init_weights,
                  scale=scale, mode=mode)


def resnet34(dims, prototypes, norm=False, init_weights=False, scale=None, mode='sce'):
    """ return a ResNet 34 object """
    return ResNet(BasicBlock, [3, 4, 6, 3], prototypes, num_classes=dims, norm=norm, init_weights=init_weights,
                  scale=scale, mode=mode)


def resnet50(dims, prototypes, norm=False, init_weights=False, scale=None, mode='sce'):
    """ return a ResNet 50 object """
    return ResNet(BottleNeck, [3, 4, 6, 3], prototypes, num_classes=dims, norm=norm, init_weights=init_weights,
                  scale=scale, mode=mode)


def resnet101(dims):
    """ return a ResNet 101 object """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=dims)


def resnet152(dims):
    """ return a ResNet 152 object """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=dims)


if __name__ == '__main__':
    prototypes = torch.from_numpy(create_max_separated_matrix.create_prototypes(10))
    # model = resnet34(dims=9, prototypes=prototypes, mode="msce")
    model = resnet34(dims=10, prototypes=prototypes, mode="sce")
    print(model)
    torchinfo.summary(model, input_size=(32, 3, 512, 512), col_names=["kernel_size",
                                                                      "input_size",
                                                                      "output_size",
                                                                      "num_params"], device="cuda:0")
