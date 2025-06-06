import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from Unet_label import UNet

# 数据集定义
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

        # 确保标签为长整型并去掉单一维度
        label = torch.squeeze(label, 0).long()

        return image, label


# 图像变换
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# 标签变换
label_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# 数据集加载
root_dir = 'cityscapes'
val_dataset = CityscapesDataset(root_dir=root_dir, split='val', transform=image_transform, target_transform=label_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# 加载模型并处理state_dict中的意外键
def load_model_weights(model, weights_path):
    state_dict = torch.load(weights_path)
    model_state_dict = model.state_dict()

    # 过滤掉多余的键
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)


# 模型初始化
deeplabv3 = deeplabv3_resnet50(pretrained=False, num_classes=34).to('cuda')
load_model_weights(deeplabv3, 'deeplabv3_cityscapes.pth')
deeplabv3.eval()

fcn = fcn_resnet50(pretrained=False, num_classes=34).to('cuda')
load_model_weights(fcn, 'fcn_cityscapes.pth')
fcn.eval()

unet = UNet().to('cuda')
unet.load_state_dict(torch.load('unet_cityscapes.pth'))
unet.eval()


# FGSM攻击函数
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    if isinstance(outputs, dict):
        outputs = outputs['out']
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = images + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 评估函数
def evaluate_model_on_adv_samples(model, adv_images, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        _, predicted = torch.max(outputs, 1)
        total = labels.nelement()
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy


# 设置epsilon
epsilon = 0.01

# 使用DeepLabV3生成对抗样本攻击FCN和UNet
sample_images, sample_labels = next(iter(val_loader))
sample_images, sample_labels = sample_images.to('cuda'), sample_labels.to('cuda')
perturbed_images_deeplab = fgsm_attack(deeplabv3, sample_images, sample_labels, epsilon)

fcn_acc_on_deeplab_adv = evaluate_model_on_adv_samples(fcn, perturbed_images_deeplab, sample_labels)
unet_acc_on_deeplab_adv = evaluate_model_on_adv_samples(unet, perturbed_images_deeplab, sample_labels)

print(f'FCN Accuracy on DeepLabV3 Adversarial Samples: {fcn_acc_on_deeplab_adv:.2f}%')
print(f'UNet Accuracy on DeepLabV3 Adversarial Samples: {unet_acc_on_deeplab_adv:.2f}%')

# 使用FCN生成对抗样本攻击DeepLabV3和UNet
perturbed_images_fcn = fgsm_attack(fcn, sample_images, sample_labels, epsilon)
deeplab_acc_on_fcn_adv = evaluate_model_on_adv_samples(deeplabv3, perturbed_images_fcn, sample_labels)
unet_acc_on_fcn_adv = evaluate_model_on_adv_samples(unet, perturbed_images_fcn, sample_labels)

print(f'DeepLabV3 Accuracy on FCN Adversarial Samples: {deeplab_acc_on_fcn_adv:.2f}%')
print(f'UNet Accuracy on FCN Adversarial Samples: {unet_acc_on_fcn_adv:.2f}%')

# 使用UNet生成对抗样本攻击DeepLabV3和FCN
perturbed_images_unet = fgsm_attack(unet, sample_images, sample_labels, epsilon)
deeplab_acc_on_unet_adv = evaluate_model_on_adv_samples(deeplabv3, perturbed_images_unet, sample_labels)
fcn_acc_on_unet_adv = evaluate_model_on_adv_samples(fcn, perturbed_images_unet, sample_labels)

print(f'DeepLabV3 Accuracy on UNet Adversarial Samples: {deeplab_acc_on_unet_adv:.2f}%')
print(f'FCN Accuracy on UNet Adversarial Samples: {fcn_acc_on_unet_adv:.2f}%')

# 对样本进行可视化
def visualize_adv_samples(original_images, perturbed_images, epsilon):
    """
    Visualize the original and perturbed images along with their differences.

    Args:
    - original_images (Tensor): Batch of original images.
    - perturbed_images (Tensor): Batch of perturbed images.
    - epsilon (float): Magnitude of perturbation.
    """
    original_images = original_images.detach().cpu()
    perturbed_images = perturbed_images.detach().cpu()

    # Create a figure to visualize the images
    plt.figure(figsize=(12, len(original_images) * 4))

    for i in range(len(original_images)):
        # Get the original, perturbed, and difference images
        orig = original_images[i].numpy().transpose(1, 2, 0)
        perturbed = perturbed_images[i].numpy().transpose(1, 2, 0)
        diff = (perturbed - orig) * 255
        diff = diff.astype('uint8')

        # Plot the original image
        plt.subplot(len(original_images), 3, i * 3 + 1)
        plt.imshow(orig)
        plt.title('Original')
        plt.axis('off')

        # Plot the perturbed image
        plt.subplot(len(original_images), 3, i * 3 + 2)
        plt.imshow(perturbed)
        plt.title('Perturbed')
        plt.axis('off')

        # Plot the difference image
        plt.subplot(len(original_images), 3, i * 3 + 3)
        plt.imshow(diff)
        plt.title(f'Difference (ε={epsilon})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
num_samples = 34
visualize_adv_samples(sample_images[:num_samples], perturbed_images_deeplab[:num_samples], epsilon)

# 可视化结果
accuracies = {
    'FCN on DeepLabV3 Adv': fcn_acc_on_deeplab_adv,
    'UNet on DeepLabV3 Adv': unet_acc_on_deeplab_adv,
    'DeepLabV3 on FCN Adv': deeplab_acc_on_fcn_adv,
    'UNet on FCN Adv': unet_acc_on_fcn_adv,
    'DeepLabV3 on UNet Adv': deeplab_acc_on_unet_adv,
    'FCN on UNet Adv': fcn_acc_on_unet_adv,
}

plt.bar(accuracies.keys(), accuracies.values())
plt.xticks(rotation=45, ha='right')
plt.xlabel('Attack Scenario')
plt.ylabel('Validation Accuracy (%)')
plt.title('Comparison of Model Performance on Adversarial Samples')
plt.show()
