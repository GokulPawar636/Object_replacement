import os
import cv2
import numpy as np

FRAME_FOLDER = "frames"
MASK_FOLDER = "seg_binary"
OUTPUT_FRAMES = "inpainted_frames"
OUTPUT_VIDEO = "inpainted_output.mp4"

os.makedirs(OUTPUT_FRAMES, exist_ok=True)

frame_files = sorted([
    f for f in os.listdir(FRAME_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

first_frame = cv2.imread(os.path.join(FRAME_FOLDER, frame_files[0]))
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20, (w, h))

prev_inpaint = None

print("🚀 High-accuracy object removal started...")

for idx, file in enumerate(frame_files):

    frame_path = os.path.join(FRAME_FOLDER, file)
    mask_path = os.path.join(MASK_FOLDER, f"mask_{idx:05d}.png")

    frame = cv2.imread(frame_path)

    if frame is None or not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, 0)

    # =========================
    # STRICT BINARY MASK
    # =========================
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # =========================
    # REMOVE NOISE
    # =========================
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # =========================
    # EXPAND MASK (controlled)
    # =========================
    kernel_big = np.ones((11,11), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel_big, iterations=1)

    # =========================
    # EDGE SOFT MASK (important)
    # =========================
    soft_mask = cv2.GaussianBlur(mask_dilated, (9,9), 0)

    # =========================
    # INPAINT ONLY MASK AREA
    # =========================
    inpainted_full = cv2.inpaint(frame, mask_dilated, 5, cv2.INPAINT_NS)

    # =========================
    # STRICT BLENDING (key fix 🔥)
    # =========================
    soft_mask_3 = soft_mask.astype(np.float32) / 255.0
    soft_mask_3 = np.stack([soft_mask_3]*3, axis=-1)

    frame_float = frame.astype(np.float32)
    inpaint_float = inpainted_full.astype(np.float32)

    # ONLY replace masked area
    result = frame_float * (1 - soft_mask_3) + inpaint_float * soft_mask_3
    result = np.clip(result, 0, 255).astype(np.uint8)

    # =========================
    # TEMPORAL SMOOTHING (safe)
    # =========================
    if prev_inpaint is not None:
        result = cv2.addWeighted(result, 0.8, prev_inpaint, 0.2, 0)

    prev_inpaint = result.copy()

    # =========================
    # SAVE
    # =========================
    cv2.imwrite(os.path.join(OUTPUT_FRAMES, f"inpaint_{idx:05d}.png"), result)
    out.write(result)

    print(f"[{idx}] done")

out.release()

print("✅ HIGH-ACCURACY VIDEO READY")
