"""
Object Mesh — General‑purpose object segmentation + Delaunay mesh
==================================================================
- Detects all non‑human objects in the first video frame using YOLOv8-seg.
- Asks you to pick which object class(es) to mask and segment.
- For every frame, extracts the masks of the selected class(es),
  computes a Delaunay triangulation over the solid interior,
  and draws a colored mesh.
- Persons are completely ignored (not detected, not listed).
"""

import subprocess
import sys
import importlib
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

# ----------------------------------------------------------------------
# Helper to install missing packages
# ----------------------------------------------------------------------
def _pip(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg.replace("-", "_").split("[")[0])
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

# Install all required packages
_pip("opencv-python", "cv2")
_pip("numpy")
_pip("streamlit")
_pip("ultralytics")      # YOLOv8 detection + segmentation
_pip("imageio-ffmpeg", "imageio_ffmpeg")
_pip("scipy")

from ultralytics import YOLO
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Object Mesh", page_icon="🎨", layout="centered")
st.title("🎨 Object Mesh")
st.markdown("Detect non‑human objects → choose which to mask → draw Delaunay mesh")
st.divider()

# ----------------------------------------------------------------------
# Load YOLOv8 segmentation model (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo():
    model_path = "yolov8m-seg.pt"
    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading YOLOv8‑seg model (~6 MB)…"):
            pass
    model = YOLO(model_path)
    return model

with st.spinner("Loading YOLOv8‑seg…"):
    yolo_model = load_yolo()
st.success("✅ Ready")
st.divider()

# ----------------------------------------------------------------------
# Settings sidebar
# ----------------------------------------------------------------------
st.subheader("⚙️ Settings")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        mesh_color_hex = st.color_picker("Mesh color", "#00AAFF")
        grid_step = st.slider("Mesh density (lower = denser)", 16, 56, 22)
        line_thick = st.slider("Line thickness", 1, 3, 1)
    with col2:
        show_nodes = st.checkbox("Show landmark nodes", True)
        output_w = st.selectbox("Output width (px)", [540, 720, 1080])

st.divider()

# ----------------------------------------------------------------------
# Upload video
# ----------------------------------------------------------------------
uploaded = st.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov", "mkv"])

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

def solid_row_fill(mask):
    """
    Convert a binary mask into a solid mask where each row is filled
    from leftmost to rightmost pixel. This removes holes for Delaunay.
    """
    h, w = mask.shape[:2]
    solid = np.zeros_like(mask)
    for y in range(h):
        nz = np.where(mask[y, :] > 0)[0]
        if len(nz) > 5:
            solid[y, nz[0]:nz[-1] + 1] = 255
    return solid


conf_thresh = st.slider("Confidence threshold", 0.1, 0.8, 0.25, 0.05)

def delaunay_mesh(edge_mask, interior_mask, extra_points, h, w, grid_step):
    """
    Build Delaunay triangulation using:
      - contour points from edge_mask (accurate shape, minimal simplification)
      - interior grid points from interior_mask (hole‑free)
      - extra_points (e.g. mask centroid)
    Returns (triangles, contour)
    """
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], None
    largest = max(contours, key=cv2.contourArea)
    # Use small epsilon to preserve shape, then decimate to at most 120 points
    simp = cv2.approxPolyDP(largest, 1.5, True)
    max_pts = 120
    if len(simp) > max_pts:
        step = max(1, len(simp) // max_pts)
        contour_pts = [(int(simp[i][0][0]), int(simp[i][0][1]))
                       for i in range(0, len(simp), step)
                       if 0 < simp[i][0][0] < w and 0 < simp[i][0][1] < h]
    else:
        contour_pts = [(int(pt[0][0]), int(pt[0][1])) for pt in simp
                       if 0 < pt[0][0] < w and 0 < pt[0][1] < h]

    interior_float = interior_mask.astype(np.float32) / 255.0
    ys, xs = np.where(interior_mask > 0)
    if len(xs) > 0:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        grid_pts = [(gx, gy)
                    for gy in range(ymin, ymax, grid_step)
                    for gx in range(xmin, xmax, grid_step)
                    if interior_float[gy, gx] > 0.5]
    else:
        grid_pts = []

    all_pts = list(set(contour_pts + grid_pts + extra_points))
    if len(all_pts) < 4:
        return [], largest

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in all_pts:
        try:
            subdiv.insert(p)
        except:
            pass
    triangles = []
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        cx = (int(x1) + int(x2) + int(x3)) // 3
        cy = (int(y1) + int(y2) + int(y3)) // 3
        if 0 < cx < w and 0 < cy < h and interior_float[cy, cx] > 0.25:
            triangles.append((int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)))
    return triangles, largest

def draw_layer(canvas, triangles, contour, color, line_thick, fill_alpha=0.15):
    """
    Draw filled triangles and outline contour using NumPy blending.
    """
    if not triangles:
        return
    if len(canvas.shape) != 3 or canvas.shape[2] != 3:
        st.warning("Canvas is not a 3‑channel BGR image – skipping draw_layer")
        return

    overlay = canvas.copy()
    fill_color = tuple(max(0, int(c * 0.07)) for c in color)

    for x1, y1, x2, y2, x3, y3 in triangles:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], fill_color)

    # NumPy blending avoids OpenCV binary_op errors
    blended = (overlay.astype(np.float32) * fill_alpha +
               canvas.astype(np.float32) * (1 - fill_alpha))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    canvas[:] = blended

    for x1, y1, x2, y2, x3, y3 in triangles:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, color, line_thick, cv2.LINE_AA)

    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, color, 2, cv2.LINE_AA)

# ----------------------------------------------------------------------
# Object detection – only non‑human classes
# ----------------------------------------------------------------------
def get_objects_from_frame(frame):
    """Run YOLOv8 on one frame and return list of detected objects (excluding persons)."""
    results = yolo_model(frame, verbose=False)[0]
    objects = []
    if results.masks is not None:
        for i, (box, mask, cls, conf) in enumerate(zip(
            results.boxes.xyxy.cpu().numpy(),
            results.masks.data.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy()
        )):
            class_name = yolo_model.names[int(cls)]
            # Skip persons entirely
            if class_name.lower() == "person":
                continue
            objects.append({
                "id": i,               # not used for selection, only for frame‑by‑frame combination
                "class": class_name,
                "confidence": float(conf),
                "bbox": box,
                "mask": mask
            })
    return objects

prev_mask = None
miss_count = 0

def process_video(video_path, selected_classes, color_bgr, grid_step, line_thick, show_nodes, out_w):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = out_w / max(orig_w, 1)
    out_w = out_w
    out_h = int(orig_h * scale) & ~1

    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(raw_out,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    prog_bar = st.progress(0, "Processing...")
    status_text = st.empty()
    start_time = time.time()
    frame_idx = 0

    prev_mask = None
    miss_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame, (out_w, out_h))

        results = yolo_model(frame, conf=conf_thresh, verbose=False)[0]

        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if results.masks is not None:
            for i, (mask, cls, conf) in enumerate(zip(
                results.masks.data.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            )):
                class_name = yolo_model.names[int(cls)]
                if class_name.lower() == "person":
                    continue
                if class_name in selected_classes:
                    bin_mask = (mask > 0.5).astype(np.uint8) * 255
                    if bin_mask.shape != combined_mask.shape:
                        bin_mask = cv2.resize(bin_mask, (combined_mask.shape[1], combined_mask.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    combined_mask = cv2.bitwise_or(combined_mask, bin_mask)

        if np.sum(combined_mask) == 0:
            if prev_mask is not None and miss_count < 3:
                combined_mask = prev_mask.copy()
                miss_count += 1
            else:
                writer.write(frame_small)
                frame_idx += 1
                prog_bar.progress(min(frame_idx / total_frames, 1.0),
                                  text=f"Frame {frame_idx}/{total_frames}")
                continue
        else:
            prev_mask = combined_mask.copy()
            miss_count = 0

        combined_mask = cv2.resize(combined_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        edge_mask = combined_mask
        interior_mask = solid_row_fill(edge_mask)

        y_coords, x_coords = np.where(edge_mask > 0)
        if len(x_coords) > 0:
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            extra_pts = [(cx, cy)]
        else:
            extra_pts = []

        tris, contour = delaunay_mesh(edge_mask, interior_mask, extra_pts,
                                      out_h, out_w, grid_step)

        canvas = frame_small.copy()
        draw_layer(canvas, tris, contour, color_bgr, line_thick, fill_alpha=0.15)

        if show_nodes:
            for (x, y) in extra_pts:
                cv2.circle(canvas, (x, y), 4, (0, 255, 180), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        writer.write(canvas)

        frame_idx += 1
        elapsed = time.time() - start_time
        fps_est = frame_idx / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_idx) / max(fps_est, 1e-3)
        prog_bar.progress(min(frame_idx / total_frames, 1.0),
                          text=f"Frame {frame_idx}/{total_frames} · {fps_est:.1f} fps · ETA {eta:.0f}s")
        if frame_idx % 30 == 0:
            status_text.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
                              use_container_width=True)

    cap.release()
    writer.release()
    prog_bar.empty()
    status_text.empty()

    final_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run([
        FFMPEG, "-y", "-i", raw_out,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-movflags", "+faststart", final_out
    ], capture_output=True, check=False)

    try:
        os.unlink(raw_out)
    except:
        pass
    return final_out, frame_idx, elapsed

# ----------------------------------------------------------------------
# Main interaction
# ----------------------------------------------------------------------
if uploaded is not None:
    st.video(uploaded)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        st.error("Could not read video file.")
    else:
        with st.spinner("Detecting objects in first frame..."):
            detected_objects = get_objects_from_frame(first_frame)

        # Get unique classes (non‑person)
        unique_classes = sorted(set(obj["class"] for obj in detected_objects))

        if not unique_classes:
            st.warning("No non‑human objects detected in the first frame. Try a different video.")
        else:
            st.subheader("🔍 Detected object classes")
            selected_classes = []
            cols = st.columns(2)
            for i, cls in enumerate(unique_classes):
                with cols[i % 2]:
                    if st.checkbox(cls, key=f"class_{cls}"):
                        selected_classes.append(cls)

            if not selected_classes:
                st.info("Select at least one object class to mask and segment.")
            else:
                if st.button("▶️ Generate Mesh Video", type="primary", use_container_width=True):
                    color_bgr = hex_bgr(mesh_color_hex)
                    final_video, frames_processed, elapsed = process_video(
                        tmp_path, selected_classes, color_bgr,
                        grid_step, line_thick, show_nodes, output_w
                    )
                    st.success(f"✅ Done – {frames_processed} frames · {elapsed:.1f}s · {frames_processed/elapsed:.1f} fps")
                    with open(final_video, "rb") as f:
                        st.download_button("📥 Download", f.read(), "object_mesh.mp4", "video/mp4", use_container_width=True)
                    try:
                        os.unlink(final_video)
                    except:
                        pass

        try:
            os.unlink(tmp_path)
        except:
            pass