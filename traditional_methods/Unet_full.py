import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. 数据集加载与预处理 (与之前版本相同)
# ----------------------------------------------------------------------
class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)
        self.images = []
        self.labels = []

        # 遍历城市文件夹，加载图像和标签路径
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            if not os.path.isdir(img_dir):
                continue
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    label_name = file_name.replace('leftImg8bit', 'gtFine_labelIds')
                    label_dir = os.path.join(self.labels_dir, city, label_name)
                    self.labels.append(label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = torch.squeeze(label, 0).long()
        return image, label

# 定义图像和标签的变换
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# 自定义标签变换类
class ToLabel:
    def __call__(self, pic):
        return torch.from_numpy(np.array(pic)).long()

label_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=Image.NEAREST), # 标签缩放必须用最近邻插值
    ToLabel()
])

# ----------------------------------------------------------------------
# 2. 完整U-Net模型定义 (核心修改部分)
# ----------------------------------------------------------------------

# U-Net的基本组件：(卷积 -> BN -> ReLU) * 2
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# U-Net的下采样模块
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

# U-Net的上采样模块
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是来自上一层（解码器部分）的特征图
        # x2 是来自对应层（编码器部分）的跳跃连接特征图
        x1 = self.up(x1)
        
        # 确保尺寸一致，防止因池化导致的尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 核心：将跳跃连接的特征图(x2)与上采样的特征图(x1)进行拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 最终的输出层
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# 完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器 (下采样路径)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器 (上采样路径)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # --- 编码器 ---
        x1 = self.inc(x)      # 保存第一层输出，用于跳跃连接
        x2 = self.down1(x1)   # 保存第二层输出
        x3 = self.down2(x2)   # 保存第三层输出
        x4 = self.down3(x3)   # 保存第四层输出
        x5 = self.down4(x4)   # 瓶颈层
        
        # --- 解码器 & 跳跃连接 ---
        x = self.up1(x5, x4)  # 上采样并与x4拼接
        x = self.up2(x, x3)   # 上采样并与x3拼接
        x = self.up3(x, x2)   # 上采样并与x2拼接
        x = self.up4(x, x1)   # 上采样并与x1拼接
        logits = self.outc(x) # 输出层
        return logits

# ----------------------------------------------------------------------
# 3. 训练设置与执行 (与之前版本基本相同)
# ----------------------------------------------------------------------

# 类别颜色映射
def decode_segmap(image, num_classes=34):
    label_colors = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
        [0, 0, 230], [119, 11, 32], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    ])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 可视化函数
def visualize(images, labels, predictions, title):
    fig, axs = plt.subplots(3, len(images), figsize=(15, 6))
    for i in range(len(images)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        label = labels[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        axs[0, i].imshow(img)
        axs[0, i].set_title('Image_Unet')
        axs[0, i].axis('off')

        axs[1, i].imshow(decode_segmap(label))
        axs[1, i].set_title('Label_Unet')
        axs[1, i].axis('off')

        axs[2, i].imshow(decode_segmap(pred))
        axs[2, i].set_title('Prediction_Unet')
        axs[2, i].axis('off')

    plt.suptitle(title)
    plt.show()

# --- 训练配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 34
model = UNet(n_channels=3, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
root_dir = './cityscapes'

# 创建数据集和数据加载器
train_dataset = CityscapesDataset(root_dir=root_dir, split='train', transform=image_transform, target_transform=label_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = CityscapesDataset(root_dir=root_dir, split='val', transform=image_transform, target_transform=label_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- 训练循环 ---
loss_list = []
print("Starting training with full U-Net...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_list.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# --- 结果展示与评估 ---
# 绘制loss曲线
plt.figure()
plt.plot(range(1, num_epochs+1), loss_list, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve (Full U-Net)')
plt.grid()
plt.show()

# 可视化分割结果
model.eval()
with torch.no_grad():
    sample_images, sample_labels = next(iter(val_loader)) # 使用验证集样本可视化
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    sample_outputs = model(sample_images)
    _, sample_predictions = torch.max(sample_outputs, 1)
    visualize(sample_images[:4], sample_labels[:4], sample_predictions[:4], title='Segmentation Results (Full U-Net)')

# 验证集评估
total, correct = 0, 0
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.nelement()
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy (Full U-Net): {accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'full_unet_cityscapes.pth')
print("Model saved to full_unet_cityscapes.pth")