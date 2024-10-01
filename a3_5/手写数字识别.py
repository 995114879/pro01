import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim



class Data_MNIST:
    def __init__(self, output_path):
        self.output_path = output_path
        self.dataset = datasets.MNIST(
            root=self.output_path,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
        self.data_loader = DataLoader(self.dataset, batch_size=4, shuffle=True)


    def train(self):
        to_img = transforms.ToPILImage()
        k = 0
        for batch_img, batch_label in self.data_loader:
            n, c, h, w = batch_img.shape
            for i in range(n):
                label = batch_label[i].item()
                output_path = Path(self.output_path + f"/MNIST/images/{label}/{k}.png")
                k += 1

                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs((os.path.dirname(output_path)))

                if c == 1:
                    img = batch_img[i].detach().numpy()
                    gray_img = (img[0] * 256).astype(np.uint8)
                    plt.imsave(output_path, gray_img, cmap='gray')
                else:
                    img = batch_img[i]
                    img = to_img(img)
                    img.save(output_path)


if __name__ == '__main__':
    data = Data_MNIST(r"../datas/MNIST")
    data.train()
