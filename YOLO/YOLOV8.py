import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import pandas as pd 

def main():
    # Define dataset paths
    root_dir = 'cityscapes_yolo_seg'
    data_yaml = os.path.join(root_dir, 'cityscapes.yaml')  # Path to your dataset configuration file

    # Initialize YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')  # Pretrained YOLOv8 segmentation model

    # Train the model
    results = model.train(
        data=data_yaml,  # Path to dataset configuration file
        epochs=10,       # Number of epochs
        imgsz=512,       # Image size
        batch=8,         # Batch size
        device=0,        # Use GPU (CUDA:0)
        project='runs/segment',  # Output directory
        name='yolov8-seg-cityscapes'  # Experiment name
    )

    # # Plot training loss curve
    # metrics = results.metrics
    # loss_list = metrics['train/seg_loss']  # Use segmentation loss

    # plt.figure()
    # plt.plot(loss_list, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('YOLOv8 Training Loss Curve')
    # plt.grid()
    # plt.show()

    # Visualize segmentation results
    best_model = YOLO('runs/segment/yolov8-seg-cityscapes/weights/best.pt')  # Load best weights

    results_csv = os.path.join('runs/segment/yolov8-seg-cityscapes', 'results.csv')
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        if 'train/seg_loss' in df.columns:
            plt.figure()
            plt.plot(df['train/seg_loss'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('YOLOv8 Training Loss Curve')
            plt.grid()
            plt.show()
        else:
            print("results.csv 中没有 'train/seg_loss' 这一列！")
    else:
        print("未找到 results.csv，训练可能未成功或路径有误。")

    # Select sample images for visualization
    val_images_root_dir = os.path.join(root_dir, 'images', 'val')  # 'cityscapes_yolo_seg/images/val'
    sample_image_paths = []
    city_folders = [os.path.join(val_images_root_dir, city) for city in os.listdir(val_images_root_dir) if os.path.isdir(os.path.join(val_images_root_dir, city))]

    for city_folder_path in city_folders:
        if len(sample_image_paths) >= 4:
            break
        for f_name in os.listdir(city_folder_path):
            if f_name.endswith(('.png', '.jpg')):
                sample_image_paths.append(os.path.join(city_folder_path, f_name))
                if len(sample_image_paths) >= 4:
                    break
    
    if not sample_image_paths:
        print(f"Warning: No sample images found in {val_images_root_dir} for visualization.")
    else:
        fig, axs = plt.subplots(1, len(sample_image_paths), figsize=(15, 6))
        if len(sample_image_paths) == 1: # Handle case for single image subplot
            axs = [axs] 
        for i, img_path in enumerate(sample_image_paths):
            img = Image.open(img_path).convert('RGB')
            results = best_model(img_path)
            seg_img = results[0].plot()  # Overlay segmentation results
            axs[i].imshow(seg_img)
            axs[i].set_title(f'Sample {i+1}')
            axs[i].axis('off')
        plt.suptitle('YOLOv8 Segmentation Results')
        plt.show()

if __name__ == '__main__':
    main()