# 完整的模型训练套路
import time
import torch
import torch.optim
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Day2.alex import AlexNet
from dataset import ImageTxtDataset

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")


# 准备数据集
def adjust_labels(labels):
    return labels % 10  # 将标签映射到0-9范围内


# 创建自定义数据集时先处理标签
class AdjustedImageTxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.original_dataset = ImageTxtDataset(txt_path, img_dir, transform)
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        return img, adjust_labels(label)


train_data = AdjustedImageTxtDataset(
    r"D:\dataset\train.txt",
    "D:\dataset\images\train",
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

test_data = torchvision.datasets.CIFAR10(
    root="../dataset_chen",
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    download=True
)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建模型 - CIFAR10有10个类别
alex = AlexNet(num_classes=10).to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(alex.parameters(), lr=learning_rate)

# 训练参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

start_time = time.time()

for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")

    # 训练步骤
    alex.train()
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        # 验证标签范围
        if torch.max(targets) >= 10 or torch.min(targets) < 0:
            print(f"非法标签值: {torch.unique(targets)}")
            continue

        outputs = alex(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}次训练的loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    print(f"本轮训练时间: {time.time() - start_time:.2f}秒")

    # 测试步骤
    alex.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = alex(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_accuracy += (predicted == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size

    print(f"测试集loss: {avg_test_loss:.4f}, 正确率: {test_accuracy:.4f}")
    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    torch.save(alex.state_dict(), f"model_save/model_{i}.pth")
    print("模型已保存")

writer.close()
print("训练完成！")