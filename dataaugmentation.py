import cv2
import albumentations as A
from albumentations.augmentations.dropout import CoarseDropout
import os
import glob
import torch
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define improved augmentation pipeline for CCTV simulation
augmentation_pipeline = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
    A.MotionBlur(blur_limit=7, p=0.5),
    A.GaussianBlur(blur_limit=7, p=0.5),
    A.GaussNoise(var_limit=(20.0, 60.0), p=0.6),
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.5),
    A.RandomRain(p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
    A.RandomShadow(p=0.3),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Paths
images_path = r"C:\Users\newfu\OneDrive\Desktop\number plates\v1\train\images"
labels_path = r"C:\Users\newfu\OneDrive\Desktop\number plates\v1\train\labels"
aug_images_path = r"C:\Users\newfu\OneDrive\Desktop\number plates\augmented data\v1\dataset\augmented_images"
aug_labels_path = r"C:\Users\newfu\OneDrive\Desktop\number plates\augmented data\v1\dataset\augmented_labels"

os.makedirs(aug_images_path, exist_ok=True)
os.makedirs(aug_labels_path, exist_ok=True)

# Get all image files
image_files = glob.glob(os.path.join(images_path, "*.jpg"))

# Augment each image
for image_path in image_files:
    image_name = os.path.basename(image_path)
    label_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + ".txt")

    # Read image using PyTorch for GPU acceleration
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_name}. Skipping...")
        continue

    height, width, _ = image.shape

    # Convert image to torch tensor and move to GPU
    image_tensor = torch.tensor(image).to(device)

    # Read bounding boxes from label file
    if not os.path.exists(label_path):
        print(f"Label file missing for {image_name}. Skipping...")
        continue

    with open(label_path, "r") as file:
        bboxes = []
        class_labels = []
        
        for line in file:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, bbox_width, bbox_height])
            class_labels.append(int(class_id))

    # Move image back to CPU for Albumentations processing
    image = image_tensor.cpu().numpy()
    
    # Apply augmentation
    try:
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_labels = augmented['class_labels']
    except Exception as e:
        print(f"Error augmenting {image_name}: {e}")
        continue

    # Save augmented image
    aug_image_path = os.path.join(aug_images_path, f"aug_{image_name}")
    cv2.imwrite(aug_image_path, augmented_image)

    # Save augmented label file
    aug_label_path = os.path.join(aug_labels_path, f"aug_{os.path.splitext(image_name)[0]}.txt")
    with open(aug_label_path, "w") as file:
        for label, bbox in zip(augmented_labels, augmented_bboxes):
            x_center, y_center, bbox_width, bbox_height = bbox
            file.write(f"{label} {x_center} {y_center} {bbox_width} {bbox_height}\n")
    
    print(f"Augmented {image_name} saved successfully.")

print("✅ Dataset augmentation for CCTV conditions completed successfully!")
