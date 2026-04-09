# import os
# import cv2
# import numpy as np
# import json
# import torch

# # =========================
# # CONFIG
# # =========================
# FRAME_FOLDER = "frames"
# MASK_FOLDER = "seg_binary"
# OUTPUT_FOLDER = "pose_output"

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Camera intrinsics (IMPORTANT for real 3D!)
# FOCAL_LENGTH = 800  # tune based on your video
# CX_OFFSET = 0.0
# CY_OFFSET = 0.0

# ALPHA = 0.6  # smoothing

# # =========================
# # LOAD MIDAS
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
# midas.to(device).eval()

# transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = transforms.dpt_transform

# print("✅ MiDaS Loaded")

# # =========================
# # TRACKING STATE
# # =========================
# state = {
#     "pos": None,
#     "vel": np.array([0, 0, 0]),
#     "rot": None,
#     "scale": None
# }

# def smooth(new, old):
#     if old is None:
#         return new
#     return ALPHA * new + (1 - ALPHA) * old

# # =========================
# # UTILS
# # =========================
# def get_stable_depth(depth_map, x, y, k=5):
#     h, w = depth_map.shape
#     x = int(np.clip(x, 0, w-1))
#     y = int(np.clip(y, 0, h-1))

#     region = depth_map[max(0,y-k):y+k, max(0,x-k):x+k]
#     return np.median(region)

# def pixel_to_world(x, y, z, w, h):
#     """Convert 2D pixel + depth → 3D camera space"""
#     X = (x - w/2) * z / FOCAL_LENGTH
#     Y = (y - h/2) * z / FOCAL_LENGTH
#     return np.array([X, -Y, z])

# # =========================
# # MAIN LOOP
# # =========================
# frame_files = sorted(os.listdir(FRAME_FOLDER))

# for idx, file in enumerate(frame_files):

#     frame = cv2.imread(os.path.join(FRAME_FOLDER, file))
#     if frame is None:
#         continue

#     h, w = frame.shape[:2]

#     # =========================
#     # DEPTH
#     # =========================
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img_rgb).to(device)

#     with torch.no_grad():
#         depth = midas(input_batch)
#         depth = torch.nn.functional.interpolate(
#             depth.unsqueeze(1),
#             size=(h, w),
#             mode="bicubic",
#             align_corners=False
#         ).squeeze().cpu().numpy()

#     depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
#     depth = 1.0 - depth  # invert

#     mask_path = os.path.join(MASK_FOLDER, f"mask_{idx:05d}.png")

#     if not os.path.exists(mask_path):
#         continue

#     mask = cv2.imread(mask_path, 0)
#     mask = cv2.medianBlur(mask, 5)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         continue

#     cnt = max(contours, key=cv2.contourArea)
#     if cv2.contourArea(cnt) < 100:
#         continue

#     # =========================
#     # CENTER (more stable)
#     # =========================
#     M = cv2.moments(cnt)
#     cx = M["m10"] / (M["m00"] + 1e-5)
#     cy = M["m01"] / (M["m00"] + 1e-5)

#     # =========================
#     # DEPTH (stable)
#     # =========================
#     z = get_stable_depth(depth, cx, cy)

#     # =========================
#     # 3D POSITION
#     # =========================
#     pos3d = pixel_to_world(cx, cy, z, w, h)

#     # =========================
#     # ORIENTATION (PCA)
#     # =========================
#     pts = cnt.reshape(-1, 2).astype(np.float32)
#     mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)

#     center = mean[0]
#     vec = eigenvectors[0]

#     roll = np.degrees(np.arctan2(vec[1], vec[0]))

#     rect = cv2.minAreaRect(cnt)
#     (_, _), (bw, bh), yaw = rect
#     if bw < bh:
#         yaw += 90

#     aspect = bh / (bw + 1e-5)
#     pitch = np.degrees(np.arctan2(aspect - 1, 1))

#     rot = np.array([yaw, pitch, roll])

#     # =========================
#     # SCALE (depth aware)
#     # =========================
#     scale = max(bw, bh) * (1 + z)

#     # =========================
#     # TEMPORAL SMOOTHING
#     # =========================
#     pos3d = smooth(pos3d, state["pos"])
#     rot = smooth(rot, state["rot"])
#     scale = smooth(scale, state["scale"])

#     state["pos"] = pos3d
#     state["rot"] = rot
#     state["scale"] = scale

#     # =========================
#     # OUTPUT
#     # =========================
#     pose = {
#         "frame": idx,
#         "position": {
#             "x": float(pos3d[0]),
#             "y": float(pos3d[1]),
#             "z": float(pos3d[2])
#         },
#         "rotation": {
#             "yaw": float(rot[0]),
#             "pitch": float(rot[1]),
#             "roll": float(rot[2])
#         },
#         "scale": float(scale)
#     }

#     with open(os.path.join(OUTPUT_FOLDER, f"pose_{idx:05d}.json"), "w") as f:
#         json.dump(pose, f, indent=4)

# print("✅ INDUSTRY-LEVEL 3D POSE GENERATED")

import os
import sys
import json
import math
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
 
import cv2
import numpy as np
 
# ── optional heavy deps (graceful degradation) ────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found – depth estimation disabled (z=0 fallback).")
 
# =============================================================================
# CONFIGURATION  (override via CLI flags or edit here)
# =============================================================================
DEFAULT_CONFIG = dict(
    # ── paths ─────────────────────────────────────────────────────────────────
    frame_folder      = "frames",
    mask_folder       = "seg_binary",
    output_folder     = "pose_output1",
 
    # ── depth model ───────────────────────────────────────────────────────────
    use_depth         = True,          
    midas_model_type  = "DPT_Large",   
 
    # ── object physical properties ────────────────────────────────────────────
    # Used for pitch estimation from 2-D projected dimensions.
    object_physical_aspect = 2.167,    
 
    # ── temporal smoothing (EMA alpha: 0=frozen, 1=no smoothing) ──────────────
    alpha_xy      = 0.55,   # centroid position
    alpha_angle   = 0.45,   # rotation  (done in complex space → no wrap)
    alpha_scale   = 0.50,   # bounding-box size
    alpha_z       = 0.35,   # depth     (slow – depth noise is high)
 
    # ── quality filters ───────────────────────────────────────────────────────
    min_mask_area = 200,    # px²  – ignore tiny noise detections
    median_blur   = 5,      # mask clean-up kernel (must be odd)
 
    # ── debug overlay ─────────────────────────────────────────────────────────
    save_debug    = True,
    debug_alpha   = 0.45,   # mask overlay transparency
)
 
 
# =============================================================================
# HELPERS
# =============================================================================
 
def load_midas(model_type: str, device):
    """Load MiDaS depth model from torch.hub."""
    print(f"[INFO] Loading MiDaS ({model_type}) …", end=" ", flush=True)
    t0 = time.time()
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.eval().to(device)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type == "MiDaS_small":
        tf = transforms.small_transform
    else:
        tf = transforms.dpt_transform
    print(f"done in {time.time()-t0:.1f}s")
    return model, tf
 
 
def estimate_depth_map(frame_rgb, midas, transform, device, h, w):
    """Return a [0,1] normalised depth map (0=near, 1=far) same size as frame."""
    inp = transform(frame_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    dm = pred.cpu().numpy()
    dm = cv2.normalize(dm, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    # MiDaS outputs INVERSE depth (larger = closer).  Invert so 1=far.
    return 1.0 - dm
 
 
def normalised_rect(cnt):
    """
    Return minAreaRect but with a STABLE angle convention:
      - rect_w  = width  of the box (shorter side)
      - rect_h  = height of the box (longer side)
      - angle   = rotation of the HEIGHT axis from the image Y-axis, in degrees
                  range: (-90, 90]
 
    OpenCV's minAreaRect convention is notoriously confusing; this wrapper
    makes it deterministic for tall objects like phones.
    """
    rect = cv2.minAreaRect(cnt)                # ((cx,cy),(w,h),angle_deg)
    (cx, cy), (rw, rh), angle = rect
 
    # ensure rh >= rw  (rh = long side = height of object)
    if rw > rh:
        rw, rh = rh, rw
        angle += 90.0
 
    # clamp to (-90, 90]
    while angle > 90.0:  angle -= 180.0
    while angle <= -90.0: angle += 180.0
 
    return (cx, cy), rw, rh, angle
 
 
def rect_to_corners(cx, cy, rw, rh, angle_deg):
    """
    Return the 4 corners of a rotated rectangle in image space.
    Order: top-left, top-right, bottom-right, bottom-left  (when angle≈0).
    This ordering is consistent with cv2.getPerspectiveTransform.
    """
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = rw / 2.0, rh / 2.0
 
    # corners relative to centre (before rotation)
    rel = np.array([[-hw, -hh],
                    [ hw, -hh],
                    [ hw,  hh],
                    [-hw,  hh]], dtype=np.float32)
 
    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
 
    corners = (rot @ rel.T).T + np.array([cx, cy])
    return corners
 
 
def corners_to_homography(corners_2d, rw, rh):
    """
    Compute a 3×3 homography H that maps the unit rectangle
        (0,0)→(0,1)→(1,1)→(1,0)  [normalised object space]
    to the four detected corners in pixel space.
 
    Use it in your engine:
        p_pixel = H @ [u, v, 1]^T   (then divide by w)
    """
    src = np.array([[0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]], dtype=np.float32)
    dst = corners_2d.astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H
 
 
def estimate_pitch(rw_px, rh_px, physical_aspect):
    """
    Estimate the pitch (tilt toward/away from camera) from the ratio of the
    projected dimensions versus the physical aspect ratio.
 
    physical_aspect = physical_height / physical_width  (e.g. 2.167 for phone)
 
    When the phone is face-on (pitch=0):
        rh_px / rw_px  ≈  physical_aspect
 
    When it tilts forward/backward by angle θ around horizontal axis:
        rh_px_projected = rh_physical * |cos θ|
        so  θ ≈ arccos( (rh_px/rw_px) / physical_aspect )
 
    Returns pitch in degrees, sign not determined (always ≥ 0).
    """
    if rw_px < 1e-3:
        return 0.0
    projected_aspect = rh_px / rw_px
    ratio = projected_aspect / physical_aspect
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(math.degrees(math.acos(ratio)))
 
 
def ema_angle(new_deg, old_complex):
    """
    Smooth an angle using complex-number EMA to avoid ±180° discontinuities.
    old_complex: previous smoothed value stored as complex number e^(i*θ).
    Returns (new_smoothed_deg, new_complex).
    """
    ALPHA = 0.45
    new_c = complex(math.cos(math.radians(new_deg)),
                    math.sin(math.radians(new_deg)))
    if old_complex is None:
        return new_deg, new_c
    smoothed_c = ALPHA * new_c + (1 - ALPHA) * old_complex
    # normalise to unit circle
    mag = abs(smoothed_c)
    if mag > 1e-9:
        smoothed_c /= mag
    smoothed_deg = math.degrees(math.atan2(smoothed_c.imag, smoothed_c.real))
    return smoothed_deg, smoothed_c
 
 
def ema(new_val, old_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * old_val
 
 
def build_pose_json(idx, cx, cy, rw, rh, angle_deg, pitch_deg, z,
                    corners, H, frame_w, frame_h):
    """
    Build the final pose dict in a format consumable by Three.js / Unity / AE.
 
    Coordinate conventions (RIGHT-HAND, Y-UP, matching Three.js / glTF):
        +X = right, +Y = up, +Z = toward camera
    """
    # ── normalised device coordinates (NDC) [-1,+1] ──────────────────────────
    ndc_x =  (cx - frame_w / 2.0) / (frame_w / 2.0)
    ndc_y = -(cy - frame_h / 2.0) / (frame_h / 2.0)   # flip Y for 3D
 
    # ── scale: convert pixel bbox to NDC units ────────────────────────────────
    scale_x = rw / frame_w   # normalised width  (0–1)
    scale_y = rh / frame_h   # normalised height (0–1)
 
    # ── camera-space Z from depth ─────────────────────────────────────────────
    # z is in [0,1] where 0=near, 1=far.  Map to a sensible world-unit range.
    # Using a [0.5 … 10.0] meter range as a reasonable default for handheld shots.
    Z_NEAR, Z_FAR = 0.3, 5.0
    z_meters = Z_NEAR + z * (Z_FAR - Z_NEAR)
 
    # ── pixel corners (for homography-based replacement) ─────────────────────
    corners_list = corners.tolist() if corners is not None else None
 
    # ── homography matrix ─────────────────────────────────────────────────────
    H_list = H.tolist() if H is not None else None
 
    # ── focal length heuristic (pinhole, f in pixels) ────────────────────────
    focal_px = frame_w        # common heuristic: f ≈ W
 
    pose = {
        "frame":          idx,
        "timestamp_ms":   idx * (1000.0 / 20.0),   # assumes 20 fps; adjust if needed
 
        # --- 2D pixel space ---------------------------------------------------
        "pixel": {
            "cx":         float(cx),
            "cy":         float(cy),
            "width":      float(rw),
            "height":     float(rh),
            "angle_deg":  float(angle_deg),   # rotation of long axis from vertical
            "corners":    corners_list,        # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        },
 
        # --- Normalised Device Coordinates (NDC) – for WebGL / Three.js -------
        "ndc": {
            "x":          float(ndc_x),       # [-1, +1]  right = +1
            "y":          float(ndc_y),       # [-1, +1]  up    = +1
            "z":          float(z),           # [0,  1]   near  =  0  (raw depth ratio)
            "scale_x":    float(scale_x),     # width  / frame_width
            "scale_y":    float(scale_y),     # height / frame_height
        },
 
        # --- 3D world space (Y-up, right-hand) --------------------------------
        "world": {
            "x":          float(ndc_x * z_meters),   # horizontal position (m)
            "y":          float(ndc_y * z_meters),   # vertical   position (m)
            "z":          float(z_meters),            # depth from camera   (m)
        },
 
        # --- Rotation (Euler, degrees) ----------------------------------------
        "rotation": {
            "yaw":        float(angle_deg),   # rotation around Y-axis (screen-plane)
            "pitch":      float(pitch_deg),   # tilt toward/away camera
            "roll":       0.0,                # not estimated (requires 3-D features)
        },
 
        # --- Homography (3×3 float matrix) ------------------------------------
        # Maps unit-rect object-space [0,1]² → pixel coords.
        # Use cv2.perspectiveTransform() or GLSL to project texture.
        "homography":     H_list,
 
        # --- Camera intrinsics (pinhole heuristic) ----------------------------
        "camera": {
            "focal_px":   float(focal_px),
            "cx":         float(frame_w / 2.0),
            "cy":         float(frame_h / 2.0),
            "width":      frame_w,
            "height":     frame_h,
        },
    }
    return pose
 
 
# =============================================================================
# DEBUG OVERLAY
# =============================================================================
 
def draw_debug(frame, mask, cx, cy, rw, rh, angle_deg, pitch_deg, z,
               corners, idx):
    out = frame.copy()
    h, w = out.shape[:2]
 
    # ── mask overlay ──────────────────────────────────────────────────────────
    if mask is not None:
        colour_mask = np.zeros_like(out)
        colour_mask[mask > 0] = (0, 200, 100)
        out = cv2.addWeighted(out, 1.0, colour_mask, 0.35, 0)
 
    # ── oriented bounding box ─────────────────────────────────────────────────
    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
 
        # draw corner labels
        labels = ["TL", "TR", "BR", "BL"]
        for i, (pt, lbl) in enumerate(zip(corners.astype(int), labels)):
            cv2.circle(out, tuple(pt), 5, (0, 255, 255), -1)
            cv2.putText(out, lbl, (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
 
    # ── centroid ──────────────────────────────────────────────────────────────
    if cx is not None:
        cv2.circle(out, (int(cx), int(cy)), 6, (0, 0, 255), -1)
 
    # ── axes (yaw arrow) ─────────────────────────────────────────────────────
    if cx is not None and angle_deg is not None:
        axis_len = int(rh / 2.5) if rh else 40
        # long axis of object
        a = math.radians(angle_deg)
        ex = int(cx + axis_len * math.sin(a))
        ey = int(cy - axis_len * math.cos(a))
        cv2.arrowedLine(out, (int(cx), int(cy)), (ex, ey),
                        (255, 80, 0), 2, tipLength=0.25)
        # cross axis
        ex2 = int(cx + (axis_len//2) * math.cos(a))
        ey2 = int(cy + (axis_len//2) * math.sin(a))
        cv2.arrowedLine(out, (int(cx), int(cy)), (ex2, ey2),
                        (80, 80, 255), 2, tipLength=0.25)
 
    # ── HUD ──────────────────────────────────────────────────────────────────
    info = [
        f"Frame : {idx:05d}",
        f"Yaw   : {angle_deg:+.1f} deg",
        f"Pitch : {pitch_deg:+.1f} deg",
        f"Depth : {z:.3f}",
        f"W x H : {rw:.0f} x {rh:.0f} px",
    ]
    pad = 10
    for i, line in enumerate(info):
        y = pad + 20 * (i + 1)
        cv2.putText(out, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
 
    # ── depth bar ────────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = w - 30, 20, 12, h - 40
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    fill = int(bar_h * z)
    cv2.rectangle(out, (bar_x, bar_y + bar_h - fill),
                  (bar_x + bar_w, bar_y + bar_h), (0, 200, 255), -1)
    cv2.putText(out, "Z", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
 
    return out
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def parse_args():
    p = argparse.ArgumentParser(description="3D Object Replacement Pose Estimator")
    p.add_argument("--frames",    default=DEFAULT_CONFIG["frame_folder"])
    p.add_argument("--masks",     default=DEFAULT_CONFIG["mask_folder"])
    p.add_argument("--output",    default=DEFAULT_CONFIG["output_folder"])
    p.add_argument("--no-depth",  action="store_true",
                   help="Disable MiDaS depth (faster, z=0)")
    p.add_argument("--no-debug",  action="store_true",
                   help="Skip saving debug overlay images")
    p.add_argument("--aspect",    type=float,
                   default=DEFAULT_CONFIG["object_physical_aspect"],
                   help="Physical height/width ratio of tracked object (default 2.167 = phone)")
    p.add_argument("--midas",     default=DEFAULT_CONFIG["midas_model_type"],
                   choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"])
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    FRAME_FOLDER   = args.frames
    MASK_FOLDER    = args.masks
    OUTPUT_FOLDER  = args.output
    USE_DEPTH      = (not args.no_depth) and TORCH_AVAILABLE
    SAVE_DEBUG     = not args.no_debug
    PHYS_ASPECT    = args.aspect
 
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
    frame_files = sorted([
        f for f in os.listdir(FRAME_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not frame_files:
        sys.exit(f"[ERROR] No image files found in {FRAME_FOLDER}")
    print(f"[INFO] Found {len(frame_files)} frames in '{FRAME_FOLDER}'")
 
    # ── depth model ───────────────────────────────────────────────────────────
    midas, midas_tf, device = None, None, None
    if USE_DEPTH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
        try:
            midas, midas_tf = load_midas(args.midas, device)
        except Exception as e:
            print(f"[WARN] Could not load MiDaS: {e}\n       Falling back to z=0.")
            USE_DEPTH = False
 
    # ── tracking state ────────────────────────────────────────────────────────
    state = {
        "cx":     None,
        "cy":     None,
        "rw":     None,   # short side (width)
        "rh":     None,   # long  side (height)
        "angle_c": None,  # complex-number for angle EMA
        "angle":  0.0,
        "z":      0.0,
    }
 
    all_poses = []
    total = len(frame_files)
 
    for idx, file in enumerate(frame_files):
        t0 = time.time()
 
        frame_path = os.path.join(FRAME_FOLDER, file)
        # mask name convention: mask_NNNNN.png  (zero-padded index)
        mask_path  = os.path.join(MASK_FOLDER, f"mask_{idx:05d}.png")
 
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARN] Cannot read frame: {frame_path}")
            continue
 
        fh, fw = frame.shape[:2]
 
        # ── depth estimation ──────────────────────────────────────────────────
        depth_map = None
        if USE_DEPTH:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = estimate_depth_map(img_rgb, midas, midas_tf, device, fh, fw)
 
        # ── load & clean mask ─────────────────────────────────────────────────
        detection_ok = False
        cx = state["cx"]
        cy = state["cy"]
        rw = state["rw"] or 1.0
        rh = state["rh"] or 1.0
        angle_deg = state["angle"]
        z = state["z"]
        mask = None
 
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
 
            # threshold to strict binary (handles semi-transparent masks)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
 
            # morphological clean-up
            k = DEFAULT_CONFIG["median_blur"]
            mask = cv2.medianBlur(mask, k)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
 
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
 
                if area >= DEFAULT_CONFIG["min_mask_area"]:
                    detection_ok = True
 
                    # ── oriented rect (FIXED convention) ──────────────────────
                    (nx, ny), nrw, nrh, new_angle = normalised_rect(cnt)
 
                    # ── depth: median over mask region ────────────────────────
                    if depth_map is not None:
                        mask_pixels = depth_map[mask > 0]
                        new_z = float(np.median(mask_pixels)) if len(mask_pixels) else 0.0
                    else:
                        new_z = 0.0
 
                    # ── temporal smoothing ────────────────────────────────────
                    cx    = ema(nx,   state["cx"],  DEFAULT_CONFIG["alpha_xy"])
                    cy    = ema(ny,   state["cy"],  DEFAULT_CONFIG["alpha_xy"])
                    rw    = ema(nrw,  state["rw"],  DEFAULT_CONFIG["alpha_scale"])
                    rh    = ema(nrh,  state["rh"],  DEFAULT_CONFIG["alpha_scale"])
                    z     = ema(new_z,state["z"],   DEFAULT_CONFIG["alpha_z"])
 
                    angle_deg, new_angle_c = ema_angle(
                        new_angle, state["angle_c"])
                    state["angle_c"] = new_angle_c
 
        # ── update state ──────────────────────────────────────────────────────
        state.update(cx=cx, cy=cy, rw=rw, rh=rh, angle=angle_deg, z=z)
 
        if cx is None:
            # no detection yet – skip
            print(f"[{idx+1:04d}/{total}] SKIP (no detection)")
            continue
 
        # ── derived quantities ────────────────────────────────────────────────
        pitch_deg  = estimate_pitch(rw, rh, PHYS_ASPECT)
        corners    = rect_to_corners(cx, cy, rw, rh, angle_deg)
        H          = corners_to_homography(corners, rw, rh)
 
        # ── build pose ────────────────────────────────────────────────────────
        pose = build_pose_json(
            idx, cx, cy, rw, rh, angle_deg, pitch_deg, z,
            corners, H, fw, fh)
        all_poses.append(pose)
 
        # ── save per-frame JSON ───────────────────────────────────────────────
        json_path = os.path.join(OUTPUT_FOLDER, f"pose_{idx:05d}.json")
        with open(json_path, "w") as f:
            json.dump(pose, f, indent=2)
 
        # ── debug overlay ─────────────────────────────────────────────────────
        if SAVE_DEBUG:
            dbg = draw_debug(frame, mask if detection_ok else None,
                             cx, cy, rw, rh, angle_deg, pitch_deg, z,
                             corners if detection_ok else None, idx)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"debug_{idx:05d}.png"), dbg)
 
        elapsed = time.time() - t0
        det_str = "✔" if detection_ok else "∅"
        print(f"[{idx+1:04d}/{total}] {det_str}  yaw={angle_deg:+.1f}°  "
              f"pitch={pitch_deg:.1f}°  z={z:.3f}  "
              f"bbox={rw:.0f}×{rh:.0f}  {elapsed:.2f}s")
 
    # ── save full sequence ────────────────────────────────────────────────────
    all_path = os.path.join(OUTPUT_FOLDER, "poses_all.json")
    with open(all_path, "w") as f:
        json.dump({"total_frames": len(all_poses), "poses": all_poses}, f, indent=2)
 
    print(f"\n✅  Done. {len(all_poses)} poses saved to '{OUTPUT_FOLDER}/'")
    print(f"    Per-frame JSON : pose_NNNNN.json")
    print(f"    Full sequence  : poses_all.json")
    if SAVE_DEBUG:
        print(f"    Debug overlays : debug_NNNNN.png")
    print()
    print("── How to use the output in your engine ──────────────────────────────")
    print("  THREE.JS  : pose.ndc.x/y/z, pose.rotation.yaw, pose.ndc.scale_x/y")
    print("  BLENDER   : pose.world.x/y/z, pose.rotation.yaw/pitch (degrees)")
    print("  HOMOGRAPHY: cv2.perspectiveTransform(pts, np.array(pose.homography))")
    print("  AE/MOCHA  : pose.pixel.corners  → 4-point track data")
 
 
if __name__ == "__main__":
    main()