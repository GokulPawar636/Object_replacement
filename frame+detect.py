import cv2
import os
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "./video/video1.mp4"
OUTPUT_FRAMES = "frames"
OUTPUT_DETECTIONS = "detections"

os.makedirs(OUTPUT_FRAMES, exist_ok=True)
os.makedirs(OUTPUT_DETECTIONS, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Tracker (for stability)
tracker = None
tracking_active = False

# =========================
# VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Save original frame
    cv2.imwrite(os.path.join(OUTPUT_FRAMES, f"frame_{frame_count:05d}.png"), frame)

    detected_frame = frame.copy()
    best_box = None
    best_conf = 0

    # =========================
    # MULTI-AUGMENT DETECTION
    # =========================
    frames_to_test = [
        frame,
        cv2.flip(frame, 1),  # horizontal flip
    ]

    for i, test_frame in enumerate(frames_to_test):

        results = model(test_frame, imgsz=1280, conf=0.25)[0]

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if cls_name == "cell phone":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # If flipped frame → convert back coordinates
                if i == 1:
                    x1 = frame_w - x2
                    x2 = frame_w - box.xyxy[0][0]

                if conf > best_conf:
                    best_conf = conf
                    best_box = (int(x1), int(y1), int(x2), int(y2))

    # =========================
    # TRACKING 
    # =========================
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        w = x2 - x1
        h = y2 - y1

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, w, h))
        tracking_active = True

    elif tracking_active:
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            best_box = (x, y, x + w, y + h)
        else:
            tracking_active = False

    # =========================
    # DRAW FINAL BOX
    # =========================
    if best_box is not None:
        x1, y1, x2, y2 = best_box

        cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detected_frame, f"Phone {best_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # Save detected frame
    cv2.imwrite(os.path.join(OUTPUT_DETECTIONS, f"det_{frame_count:05d}.png"), detected_frame)

    frame_count += 1

cap.release()

print("✅ Done! Improved detection + tracking applied")
print(f"Saved {frame_count} frames")