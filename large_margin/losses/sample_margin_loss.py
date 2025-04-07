import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleMarginLoss(nn.Module):
    def __init__(self):
        super(SampleMarginLoss, self).__init__()

    def forward(self, logits, labels):  # lebels(128,) logits(128,10)
        label_one_hot = F.one_hot(labels, logits.size()[1]).float().to(logits.device)  # (128,10)
        l1 = torch.sum(logits * label_one_hot, dim=1)  # (128,)
        tmp = logits * (1 - label_one_hot) - label_one_hot  # (128,10) 对每个样本，正确类别的logits设为-1，其他类别的logits保持不变
        l2 = torch.max(tmp, dim=1)[0]  # (128,)
        loss = l2 - l1
        return loss.mean()
