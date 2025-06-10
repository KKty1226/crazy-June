# 深度学习笔记
欠拟合：训练训练数据集表现不好，验证表现不好
过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好
## 卷积神经网络
卷积过程
```python
import torch
import torch.nn.functional as F

# 假设输入张量的形状是 [64, 3, 32, 32]
input_tensor = torch.randn(64, 3, 32, 32)

# 使用双线性插值上采样到 [224, 224]
output_tensor = F.interpolate(
    input_tensor, 
    size=(224, 224), 
    mode='bilinear', 
    align_corners=False
)

print(output_tensor.shape)  # 输出: torch.Size([64, 3, 224, 224])
```
# 图片卷积
```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)


class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


chen = CHEN()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
    writer.add_images("output", output, step)
    step += 1

```
#  tensorboard使用
TensorBoard 是 TensorFlow 提供的可视化工具，用于跟踪和可视化机器学习实验中的指标（如损失、准确率）、模型结构、权重分布、嵌入向量、图像、音频等。尽管它最初是为 TensorFlow 设计的，但现在也可以通过 torch.utils.tensorboard 支持 PyTorch

使用tensorboard命令打开
tensorboard --logdir=conv_logs

# 池化层
```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#
dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# # 最大池化没法对long整形进行池化
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype = torch.float)
# input =torch.reshape(input,(-1,1,5,5))
# print(input.shape)


class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3,
                                   ceil_mode=False)
    def forward(self,input):
        output = self.maxpool_1(input)
        return output

chen = Chen()

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = chen(imgs)
    writer.add_images("ouput",output,step)
    step += 1
writer.close()

#
# output = chen(input)
# print(output)
```