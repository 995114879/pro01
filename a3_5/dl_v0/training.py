import os

from a3_5.myutils.DataLoad import Data_Load
from a3_5.commons.metrics import Accuracy
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim


def run(
        img_path_dir
        , total_epoch=100, batch_size=8, model_output_path="./output/MNIST/dl_v0/model.pkl", use_gpu=True
):
    # 数据加载
    data = Data_Load(img_path_dir, batch_size)
    dataloader = data.load()

    # 模型构建
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.001)
    if os.path.exists(model_output_path):
        print(f"模型参数恢复：{model_output_path}")
        m = torch.load(model_output_path, map_location='cpu')
        if 'net_param' in m:
            state_dict = m['net_param']
        else:
            state_dict = m['net'].state_dict()
        print(next(net.parameters()).view(-1)[0])
        # state_dict['w'] = torch.tensor([1])
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=False)
        print(f"未进行参数迁移初始化的key列表：{missing_keys}")
        print(f"多余的参数key列表：{unexpected_keys}")
        print(next(net.parameters()).view(-1)[0])

    # 训练
    for epoch in range(total_epoch):
        # 训练阶段
        net.train()
        for _x, _y in dataloader:
            _x, _y = _x.to(device), _y.to(device)
            # 前向过程
            pred_score = net(_x)
            loss = loss_fn(pred_score, _y)
            acc = acc_fn(pred_score, _y)

            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"epoch:{epoch}/{total_epoch} train loss:{loss.item():.3f} train accuracy:{acc.item():.3f}")

            # 写入日志文件
            with open('train_log.txt', 'a') as f:
                f.write(f"epoch:{epoch}/{total_epoch} train loss:{loss.item():.3f} train accuracy:{acc.item():.3f}\n")
        # 评估阶段
        net.eval()
        with torch.no_grad():
            for _x, _y in dataloader:
                _x, _y = _x.to(device), _y.to(device)
                pred_score = net(_x)
                loss = loss_fn(pred_score, _y)
                acc = acc_fn(pred_score, _y)
                print(f"epoch:{epoch}/{total_epoch} test loss:{loss.item():.3f} test accuracy:{acc.item():.3f}")

    # 持久化
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(
        {
            # 'net': net,  # 保存模型对象
            'net_param': net.state_dict(),
            'optimizer': opt.state_dict(),
            'total_epoch': total_epoch,
            'label_2_idx': data.dataset().class_to_idx
        },
        model_output_path
    )
    print(data.dataset().class_to_idx)
