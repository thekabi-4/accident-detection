import cv2
import numpy as np
import torch
import threading
from flask import Flask, jsonify, request
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion  # For merging detections

app = Flask(__name__)

# Load Two YOLO Models (Ensure GPU Usage)
device = "cuda" if torch.cuda.is_available() else "cpu"
model1 = YOLO(r"C:\Users\newfu\OneDrive\Desktop\accident\Best modals\best models\train34\weights\best.pt").to(device)
model2 = YOLO(r"C:\Users\newfu\OneDrive\Desktop\accident\Best modals\best models\train263\weights\best.pt").to(device)

# IoU Calculation Function
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

# Function to Normalize and Denormalize Boxes
def normalize_boxes(boxes, img_w, img_h):
    return [[x1/img_w, y1/img_h, x2/img_w, y2/img_h] for x1, y1, x2, y2 in boxes]

def denormalize_boxes(boxes, img_w, img_h):
    return [[x1*img_w, y1*img_h, x2*img_w, y2*img_h] for x1, y1, x2, y2 in boxes]

# Function to Run Detection Asynchronously
def detect_async(model, frame, results_list, index):
    results = model.predict(frame, device=device, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
    boxes, scores, labels = [], [], []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(int(cls))

    results_list[index] = (boxes, scores, labels)

# Main Detection Function using 2 Models
def detect_accident(frame):
    img_h, img_w = frame.shape[:2]
    results_list = [None, None]

    # Run YOLO models in parallel threads
    thread1 = threading.Thread(target=detect_async, args=(model1, frame, results_list, 0))
    thread2 = threading.Thread(target=detect_async, args=(model2, frame, results_list, 1))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # Extract results from both models
    boxes1, scores1, labels1 = results_list[0]
    boxes2, scores2, labels2 = results_list[1]

    # Normalize boxes for WBF
    boxes1_norm = normalize_boxes(boxes1, img_w, img_h) if boxes1 else []
    boxes2_norm = normalize_boxes(boxes2, img_w, img_h) if boxes2 else []

    # Merge detections using WBF
    if boxes1_norm and boxes2_norm:
        ens_boxes_norm, ens_scores, ens_labels = weighted_boxes_fusion(
            [boxes1_norm, boxes2_norm], [scores1, scores2], [labels1, labels2], iou_thr=0.55
        )
        ens_boxes = denormalize_boxes(ens_boxes_norm, img_w, img_h)
    else:
        ens_boxes, ens_scores, ens_labels = [], [], []

    # Check for accidents using IoU threshold
    accident_detected = False
    for i in range(len(ens_boxes)):
        for j in range(i + 1, len(ens_boxes)):
            if compute_iou(ens_boxes[i], ens_boxes[j]) > 0.5:  # IoU > 0.5 means potential accident
                accident_detected = True
                x1, y1, x2, y2 = map(int, ens_boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Accident", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return accident_detected, frame

# Flask API Endpoint for Video Detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'video_path' not in request.json:
        return jsonify({"error": "Missing 'video_path' parameter"}), 400

    video_path = request.json['video_path']
    accident_detected = process_video(video_path)

    return jsonify({"accident_detected": accident_detected})

# Video Processing Function
def process_video(feed_url):
    cap = cv2.VideoCapture(feed_url)
    accident_detected = False  # Flag to track accidents in video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected, frame = detect_accident(frame)
        accident_detected = accident_detected or detected  # Persist detection across frames

        label = "Accident Detected" if detected else "No Accident"
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Accident Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return accident_detected  # Return final detection result

if __name__ == "__main__":
    feed_url = r"C:\Users\newfu\Downloads\gettyimages-948767230-640_adpp.mp4"
