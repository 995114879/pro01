
"""
用一个表格/excel统计一下所有models中已有的经典网络结构的参数量、浮点运算量、图像的平均执行时间
            参数量、浮点运算量 -->  使用库:thop
            图像的平均执行时间：针对随机产生的224*224大小的图像数据，运行100次，才中间80次的运行时间计算均值即可，每次运行需要产生不同的随机数据(提前产生)
                + 测试10000条随机数据的总耗时，需要产生batch_size=1/4/8/16
"""

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import time
from thop import profile
import numpy as np
import pandas as pd


# 自定义Dataset类
class RandomImageDataset(Dataset):
    def __init__(self, num_samples, image_size=(1, 3, 224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(*self.image_size)


# 测量单次操作的时间
def measure_single_operation_time(model, device):
    model.to(device)
    image = torch.randn(1, 3, 224, 224).to(device)
    start_time = time.time()
    with torch.no_grad():
        model(image)
    end_time = time.time()
    return end_time - start_time


# 测量处理10000条随机数据的总耗时
def measure_total_time_for_batches(model, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    start_time = time.time()
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            model(batch)
    end_time = time.time()
    return end_time - start_time


# 获取模型的参数量和浮点运算量
def get_model_params_and_flops(model, device):
    model.to(device)
    input = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(model, inputs=(input,), verbose=False)
    return params, macs


# 经典网络结构
model_names = [
    'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18',
    'resnet50', 'vgg16', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnet101',
    'wide_resnet101_2', 'resnet152', 'densenet161', 'densenet169', 'densenet201', 'mobilenet_v2', 'mobilenet_v3_large',
    'mobilenet_v3_small', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]


# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试10000条数据的总耗时的批量大小
batch_sizes = [1, 4, 8, 16]

# 结果存储
results = []

# 处理每个模型
for model_name in model_names:
    model = getattr(models, model_name)(weights=None)
    print(f"处理模型: {model_name}")

    # 计算参数量和浮点运算量
    params, flops = get_model_params_and_flops(model, device)

    # 测量单张图像执行时间
    execution_times = []
    for _ in range(100):
        execution_times.append(measure_single_operation_time(model, device))
    execution_times.sort()
    mean_execution_time = np.mean(execution_times[10:90])

    # 测量批量处理总耗时
    total_times = {}
    dataset = RandomImageDataset(10000, (3, 224, 224))
    for batch_size in batch_sizes:
        total_time = measure_total_time_for_batches(model, dataset, batch_size, device)
        total_times[batch_size] = total_time

    # 存储结果
    result = {
        'model_name': model_name,
        'params': params,
        'flops': flops,
        'mean_execution_time': mean_execution_time,
    }
    result.update({f'total_time_bs_{batch_size}': total_time for batch_size, total_time in total_times.items()})
    results.append(result)

# 将结果保存到DataFrame
df = pd.DataFrame(results)

# 将DataFrame保存到Excel文件
df.to_excel('model_performance.xlsx', index=False)

print("结果已保存到model_performance.xlsx")
