import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

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

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
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

        label = torch.squeeze(label, 0).long()  # 确保标签是长整型

        return image, label

# 定义图像和标签的变换
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# 标签变换，保持标签为整数类型
class ToLabel:
    def __call__(self, pic):
        return torch.from_numpy(np.array(pic)).long()

label_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    ToLabel()
])

# Cityscapes 标签颜色映射
label_colors = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
])

def decode_segmap(label):
    r = np.zeros_like(label).astype(np.uint8)
    g = np.zeros_like(label).astype(np.uint8)
    b = np.zeros_like(label).astype(np.uint8)

    for l in range(len(label_colors)):
        idx = label == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 设置数据集的根目录
root_dir = './cityscapes'

# 创建训练数据集和数据加载器
train_dataset = CityscapesDataset(root_dir=root_dir, split='train', transform=image_transform, target_transform=label_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 使用 DeepLabV3 模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
num_classes = 34  # Cityscapes 数据集的类别数量
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)  # 修改分类头

model = model.to('cuda')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
print('no problem')

loss_list = []  # 记录每个epoch的loss

def visualize(images, labels, predictions, title='Segmentation Result'):
    fig, axs = plt.subplots(3, len(images), figsize=(15, 6))
    for i in range(len(images)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        label = labels[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        axs[0, i].imshow(img)
        axs[0, i].set_title('Image')
        axs[0, i].axis('off')

        axs[1, i].imshow(decode_segmap(label))
        axs[1, i].set_title('Label')
        axs[1, i].axis('off')

        axs[2, i].imshow(decode_segmap(pred))
        axs[2, i].set_title('Prediction')
        axs[2, i].axis('off')

    plt.suptitle(title)
    plt.show()

# 训练循环（只保留一份）
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_list.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 绘制loss曲线
plt.figure()
plt.plot(range(1, num_epochs+1), loss_list, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid()
plt.show()

# 训练结束后可视化部分分割结果
model.eval()
with torch.no_grad():
    sample_images, sample_labels = next(iter(train_loader))
    sample_images, sample_labels = sample_images.to('cuda'), sample_labels.to('cuda')
    sample_outputs = model(sample_images)['out']
    _, sample_predictions = torch.max(sample_outputs, 1)
    visualize(sample_images[:4], sample_labels[:4], sample_predictions[:4], title='Segmentation Result')

# 验证集评估
val_dataset = CityscapesDataset(root_dir=root_dir, split='val', transform=image_transform, target_transform=label_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

total, correct = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)['out']
        _, predicted = torch.max(outputs.data, 1)
        total += labels.nelement()
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), 'deeplabv3_cityscapes.pth')