import torchinfo.torchinfo

import create_max_separated_matrix
from models.norm_layer import NormLinear, NormLinear_for_msce_LMSoftmax
from models.virtual_layer import *

__all__ = ['cnn']


class ConvBrunch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, need_relu=True):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias,
                      groups=groups),
            nn.BatchNorm2d(out_channels),
            # nn.BatchNorm2d(out_channels, affine=False),
        ]
        if need_relu:
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self, show=False, prototypes=None, num_classes=10, scale=None, init_weights=False, mode='',
                 in_channel=1):
        super(CNN, self).__init__()
        if show:
            embed_dim = 3
        else:
            embed_dim = 128
        self.in_channel = in_channel
        self.block = nn.Sequential(
            ConvBrunch(self.in_channel, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBrunch(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.in_channel == 1:
            self.fc_size = 64 * 7 * 7
        else:
            self.fc_size = 64 * 8 * 8
        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_size, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        if mode == 'msce_virtual':  # 多加一层全连接
            self.fc = nn.Linear(embed_dim, num_classes)
            self.classifier = MSCEVirtualSoftmax(prototypes)
        elif mode == 'msce_virtual_learning_strategy':
            self.fc = nn.Linear(embed_dim, num_classes)
            self.classifier = MSCEVirtualSoftmaxLearningStrategy(prototypes)
        elif mode == 'msce_resultant_virtual':
            self.fc = nn.Linear(embed_dim, num_classes)
            self.classifier = MSCEResultantVirtualSoftmax(prototypes)
        elif mode == 'msce_LMSoftmax':  # 用LMSoftmax，feature必须归一化，否则直接训练loss直接崩
            self.fc = nn.Linear(embed_dim, num_classes)
            self.classifier = NormLinear_for_msce_LMSoftmax(num_classes, num_classes + 1, prototypes)
        elif 'msce' in mode:
            self.fc = nn.Linear(embed_dim, num_classes)
            self.classifier = nn.Linear(num_classes, num_classes + 1, bias=False)
        else:  # 不多加一层全连接
            if mode in ['virtual_softmax', 'virtual_softmax_rsm', 'virtual_focal']:
                self.fc = VirtualSoftmax(embed_dim, num_classes, init_weights=init_weights)
            elif mode in ['resultant_virtual']:
                self.fc = ResultantVirtualSoftmax(embed_dim, num_classes, init_weights=init_weights, scale=scale)
            elif mode in ['virtual_learning_strategy']:
                self.fc = VirtualSoftmaxLearningStrategy(embed_dim, num_classes, init_weights=init_weights, scale=scale)
            elif mode in ['virtual_learning_strategy_addfc']:
                self.add_fc = nn.Linear(embed_dim, num_classes - 1)
                self.fc = VirtualSoftmaxLearningStrategy(num_classes - 1, num_classes, init_weights=init_weights,
                                                       scale=scale)
            elif mode in ['largest_virtual']: # added on 4-20-14 by huang
                self.fc = LargestVirtual(embed_dim, num_classes, init_weights=init_weights, scale=scale)
            else:  # 'toward_largest_margin' or 'sce'
                self.fc = nn.Linear(embed_dim, num_classes, bias=False)
                # self.fc = NormLinear(embed_dim, num_classes)  # 用sce，loss为LMSoftmax时使用

        self.scale = scale
        self.mode = mode
        if init_weights:  # 是否初始化权重
            self._initialize_weights()
        if mode == 'msce':
            with torch.no_grad():
                self.classifier.weight.data = nn.Parameter(prototypes.t())
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.get_body(x)
        if self.mode == 'sce' or self.mode == 'sce_rsm':
            output = model.fc(output)
        elif 'msce' in self.mode:
            if 'virtual' in self.mode:
                output = model.classifier(output, labels=None, mode=False)
            else:
                output = model.classifier(output)
                # output = torch.mm(embedding, prototypes)
        elif self.mode in ['virtual_softmax', 'virtual_softmax_rsm', 'virtual_focal', 'resultant_virtual',
                           'virtual_learning_strategy', 'virtual_learning_strategy_addfc']:
            output = model.fc(output, labels=None, mode=False)
        return output

    def get_body(self, x):
        x = self.block(x)
        # x = x.view(-1, self.fc_size)
        x = x.view(x.size(0), -1)
        self.fc_size = x.size(1)
        x = self.fc1(x)
        if 'addfc' in self.mode:
            x = self.add_fc(x)
        if 'msce' in self.mode:
            x = self.fc(x)
        return x

    def get_weight(self):
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


def cnn(dims, prototypes, show=False, norm=False, init_weights=False, scale=None, mode='sce', in_channel=3):
    return CNN(show=show, prototypes=prototypes, num_classes=dims, init_weights=init_weights, scale=scale, mode=mode,
               in_channel=in_channel)


if __name__ == '__main__':
    prototypes = torch.from_numpy(create_max_separated_matrix.create_prototypes(10))
    model = cnn(dims=10, prototypes=prototypes.t(), mode="sce", in_channel=3)
    print(model)
    torchinfo.summary(model, input_size=(1, 3, 32, 32), col_names=["kernel_size",
                                                                   "input_size",
                                                                   "output_size",
                                                                   "num_params"], device="cuda:1")
