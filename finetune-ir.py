import os
import cv2
import torch
from ultralytics import YOLO

# Load your base model (Replace with your base model path)
model_path = r"C:\Users\newfu\OneDrive\Desktop\accident\Best modals\best models\train12\weights\best.pt"
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directory containing your unlabeled dataset
unlabeled_data_dir = r"C:\Users\newfu\OneDrive\Desktop\accident\db\unsupervised\v3\val\Accident"
pseudo_labels_dir = r"C:\Users\newfu\OneDrive\Desktop\accident\db\usp-sp\3\Valid\labels"
pseudo_images_dir = r"C:\Users\newfu\OneDrive\Desktop\accident\db\usp-sp\3\Valid\images"

os.makedirs(pseudo_labels_dir, exist_ok=True)
os.makedirs(pseudo_images_dir, exist_ok=True)

# Adjusted thresholds to reduce false positives
confidence_threshold = 0.71  # Increased from 0.25 for stricter filtering
iou_threshold = 0.7  # Increased IoU threshold for better filtering

# Loop through all images in the dataset
for img_file in os.listdir(unlabeled_data_dir):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(unlabeled_data_dir, img_file)
        img = cv2.imread(img_path)
        height, width, _ = img.shape  # Get image dimensions
        
        # Make predictions using the base model with Test Time Augmentation (TTA)
        results = model.predict(img, conf=confidence_threshold, iou=iou_threshold, device=device, augment=True)
        
        # Extract predictions
        predictions = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        
        accident_detected = False  # Track if accident is detected in the image
        
        label_file_path = os.path.join(pseudo_labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        
        with open(label_file_path, 'w') as label_file:
            for det in predictions:
                x1, y1, x2, y2, conf, cls = det
                if conf >= confidence_threshold:  # Only consider predictions above threshold
                    accident_detected = True

                    # Convert bounding box to YOLO format (normalized)
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height

                    # Save label to .txt file
                    label_file.write(f"{int(cls)} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                    # Draw bounding box on the image for visualization (optional)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"Class {int(cls)}: {conf:.2f}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the annotated image only if an accident is detected
        if accident_detected:
            output_image_path = os.path.join(pseudo_images_dir, img_file)
            cv2.imwrite(output_image_path, img)
            print(f"✅ Pseudo-labeled image and .txt file saved: {output_image_path}")

print("🎉 Pseudo-labeling process completed!")
