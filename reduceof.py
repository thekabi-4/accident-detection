import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ✅ Load Model
best_model_path = r"C:\Users\newfu\OneDrive\Desktop\accident\runs\detect\train10\weights\best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(best_model_path)
print(f"🚀 Using device: {device}")

# ✅ Open Video Stream (Change 0 for Webcam or use a CCTV feed URL)
video_source = r"C:\Users\newfu\Downloads\gettyimages-948764164-640_adpp.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Error: Could not open video feed!")
    exit()

# ✅ Get video properties for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))  # Frames per second

# ✅ Define the output video path
output_video_path = r"C:\Users\newfu\Downloads\op.mp4"  # Change path if needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"📂 Saving detected video to: {output_video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No more frames. Exiting...")
        break

    # ✅ Run YOLO detection
    results = model.predict(frame, conf=0.4, iou=0.6, device=device)

    # ✅ Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Accident {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Show Live Feed
    cv2.imshow("Live Detection", frame)

    # ✅ Write processed frame to output video
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Detection complete! Video saved at: {output_video_path}")
