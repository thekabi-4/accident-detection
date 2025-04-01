import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO

# ✅ Load the trained YOLO model
model = YOLO(r"C:\Users\newfu\OneDrive\Desktop\accident\runs\detect\train83\weights\best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Directory to save detected accident frames and labels
output_dir = r"C:\Users\newfu\OneDrive\Desktop\accident_detected_frames"
os.makedirs(output_dir, exist_ok=True)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y1_b)
    
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def detect_accident(frame, frame_id):
    results = model.predict(frame, device=device)  # Use GPU if available
    detections = results[0].boxes  # Extract the detected boxes

    accident_detected = False
    boxes = []
    confidences = []
    classes = []

    label_data = []

    if detections is not None:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            classes.append(cls)

            if conf > 0.9:  # Lowered threshold for debugging
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the label information for this detection
            frame_height, frame_width, _ = frame.shape
            x_center = (x1 + x2) / 2 / frame_width
            y_center = (y1 + y2) / 2 / frame_height
            width = (x2 - x1) / frame_width
            height = (y2 - y1) / frame_height
            label_data.append(f"{cls} {x_center} {y_center} {width} {height}")

        # ✅ Check for overlapping boxes (Potential Accident)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if compute_iou(boxes[i], boxes[j]) > 0.5:
                    accident_detected = True
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Accident", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # ✅ Save the frame as an image file
                    frame_name = f"frame_{frame_id}.jpg"
                    label_name = f"frame_{frame_id}.txt"
                    cv2.imwrite(os.path.join(output_dir, frame_name), frame)

                    # ✅ Save the label data to a .txt file
                    with open(os.path.join(output_dir, label_name), "w") as label_file:
                        for label in label_data:
                            label_file.write(label + "\n")

    return accident_detected, frame

def process_video(feed_url):
    cap = cv2.VideoCapture(feed_url)
    if not cap.isOpened():
        print("❌ Error: Could not open video feed!")
        return

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        accident_detected, frame = detect_accident(frame, frame_id)
        label = "Accident Detected" if accident_detected else "No Accident"
        
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if accident_detected else (0, 255, 0), 2)
        cv2.imshow("Accident Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    feed_url = r"C:\Users\newfu\OneDrive\Desktop\cctv accident compilation.mp4"  # Replace with your video file path
    process_video(feed_url)
