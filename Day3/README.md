# 深度学习实战笔记

## 目录
1. [数据集预处理](#数据集预处理)
2. [数据加载实现](#数据加载实现)  
3. [GPU训练部署](#gpu训练部署)
4. [关键技巧](#关键技巧)

---

## 数据集预处理

### 1. 数据集划分脚本 `deal_with_datasets.py`
```python
# 设置随机种子
import os
import shutil
from sklearn.model_selection import train_test_split
import random
random.seed(42)

# 路径配置
dataset_dir = 'your_dataset_path'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# 创建目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 数据集划分
for class_name in os.listdir(dataset_dir):
    if class_name not in ["train", "val"]:
        class_path = os.path.join(dataset_dir, class_name)
        images = [os.path.join(class_name, f) 
                 for f in os.listdir(class_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        train_images, val_images = train_test_split(images, train_size=0.7)
        
        # 移动文件
        for img in train_images:
            shutil.move(os.path.join(dataset_dir, img), 
                       os.path.join(train_dir, img))
        for img in val_images:
            shutil.move(os.path.join(dataset_dir, img), 
                       os.path.join(val_dir, img))
        
        shutil.rmtree(class_path)
```
### 2. 生成标签文件  `prepare.py`
记得修改生成文件夹的路径
```python
import os

# 创建保存路径的函数
def create_txt_file(root_dir, txt_filename):
    # 打开并写入文件
    with open(txt_filename, 'w') as f:
        # 遍历每个类别文件夹
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # 遍历该类别文件夹中的所有图片
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")

create_txt_file(r'D:\Desktop\tcl\dataset\image2\train', 'train.txt')
create_txt_file(r'D:\Desktop\tcl\dataset\image2\val', "val.txt")
```
## 数据加载实现
自定义数据集类
```python
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageTxtDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                self.imgs_path.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img) if self.transform else img, label

# 数据增强配置
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```
## GPU训练部署
模型推送到GPU
```python
alex = AlexNet(num_classes=10).to(device)  # 模型推送到GPU
```
等价于：
```python
alex = AlexNet(num_classes=10).cuda()  # 如果确定使用GPU的替代写法
```
查看机器能否使用gpu
`python`
`import torch`
`torch.cuda.is_available()`