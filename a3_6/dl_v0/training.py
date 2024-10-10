import os
from a3_6.myutils.DataLoad import Data_Load
from a3_6.commons.metrics import Accuracy
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_file_path, tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.log_file_path = log_file_path

    def log_metrics(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        """记录训练和验证损失、准确率到 TensorBoard"""
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Loss/Validation', valid_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', valid_acc, epoch)

    def write_log(self, epoch, total_epoch, train_loss, train_acc, valid_loss, valid_acc):
        """写入日志文件"""
        with open(self.log_file_path, 'a+') as f:
            f.write(f"epoch:{epoch}/{total_epoch} train loss:{train_loss:.3f} "
                    f"train accuracy:{train_acc:.3f} valid loss:{valid_loss:.3f} "
                    f"valid accuracy:{valid_acc:.3f}\n")

    def close(self):
        """关闭 TensorBoard"""
        self.writer.close()


class ModelManager:
    def __init__(self, model_output_path, device, optimizer_cls=optim.SGD):
        self.device = device
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy()
        self.optimizer = optimizer_cls(params=self.net.parameters(), lr=0.001)
        self.model_output_path = model_output_path

    def load_model(self, load_best=False):
        """加载模型和优化器的状态"""
        model_path = self.model_output_path if not load_best else "./output/cat_vs_dog/dl_v0/best.pkl"

        if os.path.exists(model_path):
            print(f"模型参数恢复：{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('net_param')
            if state_dict is None:
                raise ValueError("Checkpoint does not contain 'net_param'.")
            self.net.load_state_dict(state_dict, strict=False)
            self.optimizer.load_state_dict(checkpoint.get('optimizer', self.optimizer.state_dict()))

    def save_model(self, model_path):
        """保存模型和优化器的状态"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'net_param': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, model_path)

    def train(self, dataloader):
        """训练模型并返回平均损失和准确率"""
        self.net.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        for batch_idx, (_x, _y) in enumerate(dataloader):
            _x, _y = _x.to(self.device), _y.to(self.device)
            pred_score = self.net(_x)
            loss = self.loss_fn(pred_score, _y)
            acc = self.acc_fn(pred_score, _y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # 打印每个 batch 的信息
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                  f"Train Loss: {loss.item():.4f}, Train Accuracy: {acc.item():.4f}")

        avg_train_loss = epoch_loss / len(dataloader)
        avg_train_acc = epoch_acc / len(dataloader)
        return avg_train_loss, avg_train_acc

    def validate(self, dataloader):
        """进行验证并返回平均损失和准确率"""
        self.net.eval()
        valid_loss, valid_acc = 0.0, 0.0
        for batch_idx, (_x, _y) in enumerate(dataloader):
            _x, _y = _x.to(self.device), _y.to(self.device)
            with torch.no_grad():
                pred_score = self.net(_x)
                loss = self.loss_fn(pred_score, _y)
                acc = self.acc_fn(pred_score, _y)

                # 打印每个验证 batch 的信息
                print(f"  Validation Batch {batch_idx + 1}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

                valid_loss += loss.item()
                valid_acc += acc.item()

        avg_valid_loss = valid_loss / len(dataloader)
        avg_valid_acc = valid_acc / len(dataloader)
        return avg_valid_loss, avg_valid_acc


def run(
        img_path_dir,
        total_epoch=100, batch_size=8,
        model_output_path="./output/cat_vs_dog/dl_v0/model.pkl",
        last_model_path="./output/cat_vs_dog/dl_v0/last.pkl",
        best_model_path="./output/cat_vs_dog/dl_v0/best.pkl",
        scripted_model_output_path="./output/cat_vs_dog/pt/model_scripted.pt",
        log_file_path="train_log.txt",
        stop_threshold=0.01,  # 损失变化阈值
        patience=5,  # 提前停止等待的epoch数
        use_gpu=True,
        tensorboard_log_dir="./tensorboard_logs",  # TensorBoard日志目录
        load_best_model=False  # 新增参数
):
    # 数据加载
    data_loader = Data_Load(img_path_dir, batch_size)
    train_loader = data_loader.load()
    valid_loader = data_loader.load(validation=True)

    # 日志记录
    logger = Logger(log_file_path, tensorboard_log_dir)

    # 模型管理
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model_manager = ModelManager(best_model_path if load_best_model else model_output_path, device)
    model_manager.load_model(load_best=load_best_model)

    # 提前停止参数
    min_loss = float('inf')
    no_improve_count = 0

    # 训练
    for epoch in range(total_epoch):
        print(f"\nEpoch [{epoch + 1}/{total_epoch}] 开始:")
        avg_train_loss, avg_train_acc = model_manager.train(train_loader)

        # 验证阶段
        print("Validation 开始:")
        avg_valid_loss, avg_valid_acc = model_manager.validate(valid_loader)

        # 将训练和验证损失、准确率记录到 TensorBoard
        logger.log_metrics(epoch, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc)

        # 写入日志
        logger.write_log(epoch + 1, total_epoch, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc)

        # 提前停止判断
        if min_loss - avg_valid_loss < stop_threshold:
            no_improve_count += 1
        else:
            min_loss = avg_valid_loss
            no_improve_count = 0
            # 保存最佳模型
            model_manager.save_model(best_model_path)

        # 每个 epoch 保存一次模型
        model_manager.save_model(last_model_path)

        if no_improve_count >= patience:
            print("损失未显著下降，提前停止训练")
            break

    # 保存最终模型为 TorchScript
    os.makedirs(os.path.dirname(scripted_model_output_path), exist_ok=True)
    scripted_model = torch.jit.script(model_manager.net)
    scripted_model.save(scripted_model_output_path)

    # 关闭 TensorBoard
    logger.close()



