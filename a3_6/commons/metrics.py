import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, y_score, y_true):
        """
        计算预测的准确率
        :param y_score: [N, C]
        :param y_true: [N]
        :return:
        """
        # 获取预测的标签值
        pred_indices = torch.argmax(y_score, dim=1)  # [N, C] --> [N]
        pred_indices = pred_indices.to(y_true.device, dtype=y_true.dtype)
        # 两者进行比较
        acc = torch.mean((pred_indices == y_true).to(dtype=torch.float))
        return acc
