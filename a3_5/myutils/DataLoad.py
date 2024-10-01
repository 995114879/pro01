

from torchvision import datasets, transforms
from torch.utils.data import DataLoader




class Data_Load:
    def __init__(self, img_path_dir, batch_size):
        self.img_path_dir = img_path_dir
        self.batch_size = batch_size

    def dataset(self):
        dataset = datasets.ImageFolder(
            root=self.img_path_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128))
            ])
        )
        return dataset

    def load(self):
        dataset = self.dataset()
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=None
        )
        return data_loader
