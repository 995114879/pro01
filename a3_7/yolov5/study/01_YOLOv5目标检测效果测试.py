import os

import torch


def t0():
    # torch.hub --> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测 --> 只要求你加载的模型文件优hubconf.py文件
    model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5:v7.0",  # 给定github上的项目名称或本地文件夹路径
        model="yolov5s",  # 给定模型文件,其实就是hubconf.py中的方法名
        source='github'  # 加载的代码/模型来源：可选：github、local
    )
    use_gpu = True
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    # 模型预测
    img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    img = r"../data/images/bus.jpg"

    # Inference 模型的推理预测
    results = model(img)

    # Results 结果展示
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()

def t1():
    print(os.path.abspath(".."))
    # torch.hub --> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测 --> 只要求你加载的模型文件优hubconf.py文件
    model = torch.hub.load(
        repo_or_dir="..",  # 给定github上的项目名称或本地文件夹路径
        model="yolov5s",  # 给定模型文件,其实就是hubconf.py中的方法名
        source='local'  # 加载的代码/模型来源：可选：github、local
    )

    # 模型预测
    # img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    img = r"../data/images/bus.jpg"

    # Inference 模型的推理预测
    results = model(img)

    # Results 结果展示
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()

if __name__ == '__main__':
    # t0()
    t1()    
