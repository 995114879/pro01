import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Data_Load:
    def __init__(self, img_path_dir, batch_size, valid_split=0.2):
        self.img_path_dir = img_path_dir
        self.batch_size = batch_size
        self.valid_split = valid_split

        # 检查数据集目录
        if not os.path.exists(img_path_dir):
            raise FileNotFoundError(f"数据集目录 {img_path_dir} 不存在，请检查路径。")

        # 设置数据转换
        self.train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转
            transforms.ToTensor()
        ])
        self.valid_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def dataset(self, train=True):
        transform = self.train_transform if train else self.valid_transform
        dataset = datasets.ImageFolder(root=self.img_path_dir, transform=transform)
        return dataset

    def load(self, validation=False):
        dataset = self.dataset(train=not validation)

        if validation:
            valid_size = int(len(dataset) * self.valid_split)
            train_size = len(dataset) - valid_size
            train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
            dataset = valid_dataset if validation else train_dataset

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not validation,
            num_workers=0,
            collate_fn=None
        )
        return data_loader
