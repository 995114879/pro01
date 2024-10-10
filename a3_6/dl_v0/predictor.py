import os.path
from io import BytesIO
from typing import Optional

import requests
import base64
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader


class Predictor:
    def __init__(self, algo_path, use_gpu=True):
        super(Predictor, self).__init__()

        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            algo_result = torch.load(algo_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            algo_result = torch.load(algo_path, map_location="cpu")

        if 'net_param' in algo_result:
            state_dict = algo_result['net_param']
        else:
            state_dict = algo_result['net'].state_dict()
        net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=False)
        if len(missing_keys) > 0:
            raise ValueError(f"部分参数未初始化，不能进行推理：{missing_keys}")
        self.net = net.eval()
        self.label2idx = algo_result['label_2_idx']
        self.idx2label = {}
        for label, idx in self.label2idx.items():
            self.idx2label[idx] = label
        print(self.label2idx)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])

    def load_img(self,
                 img_path: Optional[str] = None,
                 img_url: Optional[str] = None,
                 base64_img: Optional[str] = None):
        if base64_img:
            print("直接恢复base64编码的图像数据")
            img_content = base64.b64decode(base64_img)
            return 0, Image.open(BytesIO(img_content))
        elif img_url:
            print(f"从网络上下载图像进行预测：{img_url}")
            response = requests.get(img_url)
            if response.status_code == 200:
                return 1, Image.open(BytesIO(response.content))
            else:
                print(f"从网络上加载图片失败：{response.status_code} - {img_url}")
                return -1, None
        elif img_path:
            print(f"加载服务器本地图像进行预测：{img_path}")
            if not os.path.exists((img_path)):
                return -1, None
            return 2, default_loader(img_path)
        else:
            return -1, None

    @torch.no_grad()
    def predict(self,
                img_path: Optional[str] = None,
                img_url: Optional[str] = None,
                base64_img: Optional[str] = None,
                k=1):
        """
        基于给定路径进行预测，产生返回结果
        :param img_path:
        :param img_url:
        :param base64_img:
        :param k:获取前topk
        :return:返回一个字典对象
        """
        # 1. 图像数据加载并转换
        flag, img = self.load_img(img_path, img_url, base64_img)
        if img is None:
            return {'code': '1', 'msg': f"无法加载有效图像 --> img_url:{img_url}  img_path:{img_path}"}
        if flag == 0:
            img_url = None
            img_path = None
        elif flag == 1:
            img_path = None
        else:
            img_url = None
        x = self.transform(img)  # tensor [C,H,W]

        # 2. 模型预测
        y_score = self.net(x[None])[0]  # [C,H,W] --> [1,C,H,W] --> [1,17] --> [17]
        y_proba = torch.softmax(y_score, dim=0).numpy()
        y_proba_idx = list(zip(y_proba, range(len(y_proba))))
        y_proba_idx = sorted(y_proba_idx, reverse=True, key=lambda t: t[0])
        print(y_proba_idx)

        # 3. 获取预测类别值和概率值
        k = min(max(k, 1), len(y_proba_idx))
        result = {
            'code': 0,
            'topk': k,
            'datas': [],
            'flag': flag,
            'image_path': img_path,
            'img_url': img_url
        }
        for proba, label_idx in y_proba_idx:
            r = {
                'label': self.idx2label[label_idx], 'label_idx': label_idx, 'proba': f"{proba:.2f}"
            }
            result['datas'].append(r)
            k -= 1
            if k <= 0:
                break
        return result
