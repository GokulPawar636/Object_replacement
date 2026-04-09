import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# =========================
# SETTINGS
# =========================
FRAME_FOLDER = "frames"
OUTPUT_MASK_FOLDER = "seg_binary"
OUTPUT_OVERLAY_FOLDER = "seg_output"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_OVERLAY_FOLDER, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
yolo_model = YOLO("yolov8x.pt")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# =========================
# TRACKING MEMORY
# =========================
last_box = None  # store last detected box

# =========================
# LOAD FRAMES
# =========================
frame_files = sorted(os.listdir(FRAME_FOLDER))

for idx, file in enumerate(frame_files):

    path = os.path.join(FRAME_FOLDER, file)
    frame = cv2.imread(path)

    if frame is None:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # =========================
    # YOLO DETECTION
    # =========================
    results = yolo_model(frame, imgsz=1280, conf=0.25)[0]

    final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    overlay = frame.copy()

    predictor.set_image(rgb_frame)

    detected = False

    # =========================
    # FIND PHONE
    # =========================
    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        if cls_name == "cell phone" and conf > 0.25:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            input_box = np.array([x1, y1, x2, y2])

            last_box = input_box  # save for tracking
            detected = True
            break

    # =========================
    # FALLBACK (TRACKING)
    # =========================
    if not detected and last_box is not None:
        input_box = last_box
    elif last_box is None:
        # nothing to track yet
        cv2.imwrite(
            os.path.join(OUTPUT_MASK_FOLDER, f"mask_{idx:05d}.png"),
            final_mask
        )
        continue

    # =========================
    # SAM SEGMENTATION
    # =========================
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=True
    )

    mask = masks[np.argmax(scores)]
    mask = (mask * 255).astype(np.uint8)

    # =========================
    # SMOOTH MASK
    # =========================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    final_mask = mask

    # =========================
    # SAVE MASK
    # =========================
    cv2.imwrite(
        os.path.join(OUTPUT_MASK_FOLDER, f"mask_{idx:05d}.png"),
        final_mask
    )

    # =========================
    # OVERLAY
    # =========================
    colored = np.zeros_like(frame)
    colored[:, :, 1] = final_mask

    overlay = cv2.addWeighted(frame, 1, colored, 0.5, 0)

    cv2.imwrite(
        os.path.join(OUTPUT_OVERLAY_FOLDER, f"overlay_{idx:05d}.png"),
        overlay
    )

print("✅ DONE — Continuous segmentation across all frames")