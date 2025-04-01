import time
from ultralytics import YOLO
import torch
import os
import glob

def get_latest_checkpoint(run_path):
    """Finds the latest checkpoint file in the training directory."""
    checkpoint_pattern = os.path.join(run_path, "weights", "last.pt")
    if os.path.exists(checkpoint_pattern):
        return checkpoint_pattern
    return None

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Define paths
    base_model_path = "yolo11l.pt"
    best_weights = r"C:\Users\newfu\OneDrive\Desktop\accident\Best modals\best models\train12\weights\best.pt"
    runs_dir = r"runs\train"

    # Find the latest training run
    existing_runs = sorted(glob.glob(os.path.join(runs_dir, "train*")), key=os.path.getmtime, reverse=True)
    latest_run = existing_runs[0] if existing_runs else None
    last_checkpoint = get_latest_checkpoint(latest_run) if latest_run else None

    # Load Model
    if last_checkpoint:
        print(f"✅ Resuming training from: {last_checkpoint}")
        model = YOLO(last_checkpoint)
    elif os.path.exists(best_weights):
        print(f"✅ Loading pre-trained model from: {best_weights}")
        model = YOLO(best_weights)
    else:
        print("⚠️ Warning: No previous training found. Starting from base model.")
        model = YOLO(base_model_path)

    model.to(device)

    # Dataset
    dataset = r"C:\Users\newfu\OneDrive\Desktop\accident\db\usp-sp\3\data.yaml"

    # Training Parameters
    total_epochs = 20
    batch_size = 8
    img_size = 640
    patience = 5  # Early stopping patience
    cooldown_time = 0  # 3-minute cooldown

    print(f"🚀 Starting training on {device}...\n")

    best_map50 = 0.0  # Track best validation mAP@50
    no_improvement_epochs = 0  # Track epochs without improvement

    for epoch in range(1, total_epochs + 1):
        print(f"\n🛠️ Epoch {epoch}/{total_epochs} - Training on dataset: {dataset}")

        model.train(
            data=dataset,
            epochs=1,  # Train one epoch at a time
            batch=batch_size,
            imgsz=img_size,
            device=device,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.001,  # Stronger L2 regularization
            iou=0.65,  # Lower IoU to improve generalization
            conf=0.45,  # Reduce confidence threshold slightly
            augment=True,
            workers=2,
            amp=True,
            cache="disk",
            hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
            fliplr=0.6,
            scale=0.6,
            mixup=0.7,
            mosaic=0.5,  # Reduce mosaic augmentation to avoid overfitting
            resume=bool(last_checkpoint)
        )

        print(f"\n📊 Validating after epoch {epoch}")
        results = model.val(data=dataset)
        mAP50 = results.box.map50  # Extract mAP@50 score

        print(f"📈 Validation mAP@50: {mAP50:.4f}")

        # Save validation results to a log file
        with open("validation_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch}, mAP@50: {mAP50:.4f}\n")

        # Check for improvement
        if mAP50 > best_map50:
            print("✅ New best model found! Saving checkpoint...")
            best_map50 = mAP50
            no_improvement_epochs = 0  # Reset patience counter
            
            best_model_path = "best_model.pt"
            model.save(best_model_path)
            print(f"✅ Best model saved as {best_model_path}")
        else:
            no_improvement_epochs += 1
            print(f"⚠️ No improvement for {no_improvement_epochs}/{patience} epochs.")

        # Early stopping condition
        if no_improvement_epochs >= patience:
            print("🛑 Early stopping triggered! Training stopped.")
            break

        print(f"⏳ Cooldown for {cooldown_time // 60} minutes...\n")
        time.sleep(cooldown_time)

    print("🎉 Training completed!")

if __name__ == "__main__":
    train_model()
