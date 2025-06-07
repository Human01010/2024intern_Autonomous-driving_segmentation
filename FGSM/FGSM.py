import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. 数据集加载与预处理 
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

# 2. 完整U-Net模型定义
class DoubleConv(nn.Module):
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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# 3. 对抗攻击与训练设置 
# FGSM攻击函数 (从 FGSM_final.py 引入)
def fgsm_attack(model, images, labels, epsilon):
    """
    对给定的图像批量生成FGSM对抗样本
    """
    # 开启图像的梯度计算
    images.requires_grad = True
    
    # 前向传播，计算损失
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # 清空旧的梯度
    model.zero_grad()
    # 反向传播，计算图像的梯度
    loss.backward()
    
    # 收集梯度数据
    data_grad = images.grad.data
    
    # 获取梯度的符号
    sign_data_grad = data_grad.sign()
    
    # 创建扰动图像
    perturbed_image = images + epsilon * sign_data_grad
    
    # 将图像像素值限制在[0, 1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # 返回扰动后的图像
    return perturbed_image

# 类别颜色映射
label_colors = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
])
def decode_segmap(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(len(label_colors)):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# --- 训练配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 34
model = UNet(n_channels=3, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 建议使用稍小的学习率
num_epochs = 10
epsilon = 0.02 # FGSM扰动大小，可以调整
root_dir = './cityscapes'

# 创建数据集和数据加载器
train_dataset = CityscapesDataset(root_dir=root_dir, split='train', transform=image_transform, target_transform=label_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # 建议减小batch size以防显存不足
val_dataset = CityscapesDataset(root_dir=root_dir, split='val', transform=image_transform, target_transform=label_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --- 对抗训练循环 ---
loss_list = []
print("Starting Adversarial Training with full U-Net...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 1. 生成对抗样本
        # 注意：这里模型需要处于评估模式（eval）来生成稳定的梯度，但优化时需要处于训练模式（train）
        model.eval() # 切换到评估模式以生成对抗样本
        perturbed_images = fgsm_attack(model, images, labels, epsilon)
        model.train() # 切换回训练模式以进行参数更新

        # 2. 在对抗样本上进行训练
        optimizer.zero_grad()
        # 将生成的对抗样本（已在GPU上）送入模型
        outputs = model(perturbed_images) 
        # 使用原始标签计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_list.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Adversarial Loss: {epoch_loss:.4f}')

# --- 结果展示与评估 ---
# 绘制loss曲线
plt.figure()
plt.plot(range(1, num_epochs+1), loss_list, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Adversarial Loss')
plt.title('Adversarial Training Loss Curve (Full U-Net)')
plt.grid()
plt.show()

# --- 验证集评估 (在干净数据和对抗数据上) ---
model.eval()
correct_clean, total_clean = 0, 0
correct_adv, total_adv = 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        # 1. 在干净数据上评估
        outputs_clean = model(images)
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        total_clean += labels.nelement()
        correct_clean += (predicted_clean == labels).sum().item()
        
        # 2. 生成对抗数据并评估
        # 这里需要开启梯度计算来生成对抗样本
        adv_images = fgsm_attack(model, images, labels, epsilon).detach()
        outputs_adv = model(adv_images)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        total_adv += labels.nelement()
        correct_adv += (predicted_adv == labels).sum().item()

accuracy_clean = 100 * correct_clean / total_clean
accuracy_adv = 100 * correct_adv / total_adv

print(f'Validation Accuracy on Clean Data: {accuracy_clean:.2f}%')
print(f'Validation Accuracy on Adversarial Data (epsilon={epsilon}): {accuracy_adv:.2f}%')

# --- 可视化对抗样本及其分割结果 ---
def visualize_adversarial(original_img, perturbed_img, label, prediction, title):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    original_img = original_img.cpu().numpy().transpose(1, 2, 0)
    perturbed_img = perturbed_img.cpu().numpy().transpose(1, 2, 0)
    label = label.cpu().numpy()
    prediction = prediction.cpu().numpy()

    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(perturbed_img)
    axs[1].set_title('Perturbed (Adversarial) Image')
    axs[1].axis('off')

    axs[2].imshow(decode_segmap(label))
    axs[2].set_title('Ground Truth Label')
    axs[2].axis('off')
    
    axs[3].imshow(decode_segmap(prediction))
    axs[3].set_title('Prediction on Perturbed Image')
    axs[3].axis('off')

    plt.suptitle(title)
    plt.show()

# 从验证集中取一个批次进行可视化
model.eval()
with torch.no_grad():
    sample_images, sample_labels = next(iter(val_loader))
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)

    # 生成对抗样本
    perturbed_sample_images = fgsm_attack(model, sample_images, sample_labels, epsilon)
    
    # 获取模型在对抗样本上的预测
    sample_outputs = model(perturbed_sample_images)
    _, sample_predictions = torch.max(sample_outputs, 1)

    # 可视化前两个样本
    for i in range(2):
        visualize_adversarial(
            sample_images[i],
            perturbed_sample_images[i],
            sample_labels[i],
            sample_predictions[i],
            title=f'Adversarial Example {i+1} (epsilon={epsilon})'
        )

# 保存经过对抗训练的模型
torch.save(model.state_dict(), 'unet_adversarial_trained.pth')
print("Adversarially trained model saved to unet_adversarial_trained.pth")