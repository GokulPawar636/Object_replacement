"""
╔══════════════════════════════════════════════════════════════╗
║     REALISTIC PHONE REPLACEMENT SYSTEM — iPhone 14 Pro      ║
║  Input : phone.mp4  +  iphone_14_pro.zip (GLTF + Textures)  ║
║  Output: output_replaced.mp4                                 ║
╚══════════════════════════════════════════════════════════════╝

TECHNOLOGY STACK
────────────────
• Phone Detection   : GrabCut (OpenCV) — no model download needed
• Corner Tracking   : Shi-Tomasi corners + Lucas-Kanade Optical Flow
• Texture Extraction: Direct PBR UV-map crop from GLTF baseColor texture
• Perspective Warp  : 4-point homography (cv2.getPerspectiveTransform)
• Realistic Blend   : Poisson seamless cloning + brightness/contrast match
• Edge Quality      : Feathered alpha mask with Gaussian blur
• Frame Stabilise   : Kalman-smoothed corner trajectories

REQUIREMENTS
────────────
    pip install opencv-python numpy Pillow streamlit

RUN
───
    streamlit run phone_replacement_app.py
"""

import os, sys, zipfile, tempfile, time
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="iPhone 14 Pro Replacement",
    page_icon="📱",
    layout="centered",
)

st.markdown("""
<style>
    .main { background: #0e1117; }
    h1 { color: #f5f5f7; letter-spacing: -1px; }
    .stProgress > div > div { background: linear-gradient(90deg,#2997ff,#30d158); }
    .info-box { background:#1c1c1e; border-radius:12px; padding:16px;
                border-left:4px solid #2997ff; margin:8px 0; color:#ebebf5; }
</style>
""", unsafe_allow_html=True)

st.title("📱 Realistic iPhone 14 Pro Replacement")
st.markdown(
    '<div class="info-box">Upload your video and the iPhone 14 Pro ZIP to replace '
    'the phone in every frame with a photorealistic iPhone 14 Pro composite.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    output_w = st.selectbox("Output resolution (width)", [540, 720, 800, 1080], index=2)

    blend_mode = st.radio(
        "Blending mode",
        ["Feather (fast)", "Poisson (realistic)"],
        index=1,
    )
    use_poisson = blend_mode == "Poisson (realistic)"

    smooth_tracking = st.checkbox("Smooth corner tracking (Kalman)", value=True)
    match_brightness = st.checkbox("Match iPhone brightness to scene", value=True)
    add_reflection = st.checkbox("Add subtle screen reflection", value=True)

    st.markdown("---")
    st.caption("💡 Tip: Poisson blending gives the most realistic result but is ~2× slower.")

# ─────────────────────────────────────────────────────────────────────────────
#  FILE UPLOADERS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("📹 Upload Video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
with col2:
    zip_file = st.file_uploader("📦 Upload iPhone ZIP (with GLTF textures)", type=["zip"])

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: EXTRACT iPHONE FRONT FACE FROM ZIP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_iphone_texture(zip_bytes: bytes) -> np.ndarray:
    """
    Extract the iPhone 14 Pro front-face PNG from the GLTF ZIP.

    The ZIP contains a 4096×4096 PBR baseColor texture.
    The front face (portrait, Dynamic Island visible) sits in the UV region:
        x: 2580 – 3850,  y: 0 – 2870
    This was determined by cross-referencing the emissive map (screen glow)
    with the baseColor texture to find the exact screen region.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "model.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # Look for the baseColor texture (highest priority)
        texture_path = None
        priority_names = ["material_basecolor", "basecolor", "diffuse", "albedo"]
        for root, _, files in os.walk(tmpdir):
            for fn in files:
                if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fn_lower = fn.lower().replace(" ", "_").replace("-", "_")
                for p in priority_names:
                    if p in fn_lower:
                        texture_path = os.path.join(root, fn)
                        break
                if texture_path:
                    break

        # Fallback: largest image file
        if texture_path is None:
            candidates = []
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        fp = os.path.join(root, fn)
                        candidates.append((os.path.getsize(fp), fp))
            if candidates:
                texture_path = max(candidates)[1]

        if texture_path is None:
            return None

        pil_img = Image.open(texture_path).convert("RGBA")
        tw, th = pil_img.size

        # ── Crop the front face region from the UV map ─────────────────────
        # For a 4096×4096 map: front face is at x=2580-3850, y=0-2870
        # Scale proportionally if texture is a different resolution
        if tw >= 3000 and th >= 2800:
            x0 = int(tw * 2580 / 4096)
            x1 = int(tw * 3850 / 4096)
            y0 = 0
            y1 = int(th * 2870 / 4096)
        else:
            # Fallback: use entire right half (landscape textures, etc.)
            x0, y0, x1, y1 = tw // 2, 0, tw, th

        front_face = pil_img.crop((x0, y0, x1, y1))

        # Compose onto black background (remove alpha artifacts)
        bg = Image.new("RGB", front_face.size, (0, 0, 0))
        bg.paste(front_face.convert("RGB"), mask=front_face.split()[3])
        return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: DETECT PHONE CORNERS IN FIRST FRAME  (GrabCut + Contour)
# ─────────────────────────────────────────────────────────────────────────────
def detect_phone_corners(frame: np.ndarray) -> np.ndarray | None:
    """
    Returns 4 corner points of the phone in the frame as float32 array.
    Order: top-left, top-right, bottom-right, bottom-left.

    Algorithm:
      1. Run GrabCut with a loose central rectangle (phone typically center-frame)
      2. Clean the resulting mask with morphological ops
      3. Find the largest contour → fit minAreaRect → get 4 corner points
      4. Order corners TL → TR → BR → BL
    """
    h, w = frame.shape[:2]

    # ── GrabCut initialization ─────────────────────────────────────────────
    mask = np.zeros((h, w), np.uint8)

    # Seed: assume phone occupies the central ~60% of frame width
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.04)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    # ── Morphological cleanup ──────────────────────────────────────────────
    k = np.ones((9, 9), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour with phone-like aspect ratio
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w * 0.05):       # must be at least 5% of frame
            continue
        rr = cv2.minAreaRect(c)
        rw, rh = rr[1]
        if rw == 0 or rh == 0:
            continue
        asp = max(rw, rh) / min(rw, rh)
        if 1.3 <= asp <= 3.2 and area > best_area:
            best = rr
            best_area = area

    if best is None:
        # Fallback: largest contour regardless of aspect
        c = max(contours, key=cv2.contourArea)
        best = cv2.minAreaRect(c)

    # ── Get 4 corners and order them ───────────────────────────────────────
    box = cv2.boxPoints(best).astype(np.float32)
    return _order_corners(box)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: TL, TR, BR, BL."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # TL: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # BR: largest  x+y
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # TR: smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # BL: largest  y-x
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: KALMAN SMOOTHER FOR CORNER COORDINATES
# ─────────────────────────────────────────────────────────────────────────────
class CornerKalman:
    """Simple 1-D Kalman filter per coordinate (8 scalars for 4 corners)."""
    def __init__(self, initial: np.ndarray):
        n = initial.size
        self.kf = cv2.KalmanFilter(n * 2, n)          # state=[pos,vel], meas=[pos]
        self.kf.transitionMatrix = np.eye(n * 2, dtype=np.float32)
        for i in range(n):
            self.kf.transitionMatrix[i, n + i] = 1.0  # pos += vel
        self.kf.measurementMatrix = np.zeros((n, n * 2), dtype=np.float32)
        for i in range(n):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov     = np.eye(n * 2, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(n,     dtype=np.float32) * 5e-2
        self.kf.errorCovPost        = np.eye(n * 2, dtype=np.float32)
        state = np.zeros(n * 2, dtype=np.float32)
        state[:n] = initial.flatten()
        self.kf.statePost = state.reshape(-1, 1)
        self.n = n

    def update(self, corners: np.ndarray) -> np.ndarray:
        self.kf.predict()
        meas = corners.flatten().astype(np.float32).reshape(-1, 1)
        self.kf.correct(meas)
        smoothed = self.kf.statePost[:self.n].reshape(4, 2)
        return smoothed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: OPTICAL FLOW CORNER TRACKING
# ─────────────────────────────────────────────────────────────────────────────
def track_corners_optflow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_corners: np.ndarray,
) -> np.ndarray | None:
    """
    Track 4 corners from prev frame to curr frame using LK optical flow.
    Returns updated 4 corners, or None if tracking fails.
    """
    pts = prev_corners.reshape(4, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if next_pts is None or status is None:
        return None
    if status.sum() < 3:
        return None
    return next_pts.reshape(4, 2).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: COMPOSITE — WARP + BLEND iPhone ONTO FRAME
# ─────────────────────────────────────────────────────────────────────────────
def composite_iphone(
    frame:          np.ndarray,
    iphone_img:     np.ndarray,
    dst_corners:    np.ndarray,
    use_poisson:    bool = True,
    match_brightness: bool = True,
    add_reflection: bool = True,
) -> np.ndarray:
    """
    Perspective-warp the iPhone texture onto the frame at dst_corners.

    Steps:
      1. Build homography: iPhone → dst_corners
      2. Warp iPhone into frame space
      3. (optional) Brightness-match iPhone to scene lighting
      4. (optional) Add subtle Fresnel-style screen reflection
      5. Blend using Poisson seamless clone or feathered alpha mask
    """
    fh, fw = frame.shape[:2]
    ih, iw = iphone_img.shape[:2]

    src_corners = np.float32([
        [0,  0 ],
        [iw, 0 ],
        [iw, ih],
        [0,  ih],
    ])

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # ── Warp iPhone ─────────────────────────────────────────────────────────
    warped = cv2.warpPerspective(iphone_img, M, (fw, fh),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    # ── Build polygon mask ───────────────────────────────────────────────────
    poly = dst_corners.astype(np.int32)
    hard_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillConvexPoly(hard_mask, poly, 255)

    # ── Brightness matching ──────────────────────────────────────────────────
    if match_brightness:
        # Sample mean brightness of the scene in the phone region (background)
        scene_roi = cv2.bitwise_and(frame, frame, mask=hard_mask)
        iphone_roi = cv2.bitwise_and(warped, warped, mask=hard_mask)
        scene_mean  = float(cv2.mean(cv2.cvtColor(scene_roi, cv2.COLOR_BGR2GRAY), mask=hard_mask)[0])
        iphone_mean = float(cv2.mean(cv2.cvtColor(iphone_roi, cv2.COLOR_BGR2GRAY), mask=hard_mask)[0])
        if iphone_mean > 1:
            scale = np.clip(scene_mean / iphone_mean, 0.55, 1.45)
            warped = np.clip(warped.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # ── Screen reflection overlay ────────────────────────────────────────────
    if add_reflection:
        # Warp a soft white gradient into the phone region → simulates glass reflection
        refl = np.zeros_like(warped, dtype=np.float32)
        # Diagonal gradient top-left corner
        for c in range(3):
            refl[:, :, c] = np.fromfunction(
                lambda y, x: 255 * np.exp(-((x / fw) ** 2 + (y / fh) ** 2) * 4),
                (fh, fw), dtype=np.float32,
            )
        refl = cv2.warpPerspective(refl, M, (fw, fh))
        alpha_refl = 0.07   # subtle: 7% white highlight
        warped_f = warped.astype(np.float32)
        mask_f   = (hard_mask[:, :, None] / 255.0)
        warped_f = np.clip(warped_f + refl * alpha_refl * mask_f, 0, 255)
        warped   = warped_f.astype(np.uint8)

    # ── Blend ────────────────────────────────────────────────────────────────
    if use_poisson:
        # Poisson seamless cloning (OpenCV) — most realistic
        # Center of the destination polygon
        cx = int(dst_corners[:, 0].mean())
        cy = int(dst_corners[:, 1].mean())
        # Clamp center to valid range
        cx = max(1, min(cx, fw - 2))
        cy = max(1, min(cy, fh - 2))
        try:
            result = cv2.seamlessClone(warped, frame, hard_mask, (cx, cy),
                                        cv2.NORMAL_CLONE)
        except cv2.error:
            # Fallback to feather blend if Poisson fails
            result = _feather_blend(frame, warped, hard_mask)
    else:
        result = _feather_blend(frame, warped, hard_mask)

    return result


def _feather_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 6,
) -> np.ndarray:
    """Soft-edge alpha blend."""
    soft = cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), feather_px)
    a = soft[:, :, None].astype(np.float32) / 255.0
    result = (background.astype(np.float32) * (1 - a) +
              foreground.astype(np.float32) * a)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    video_path:       str,
    iphone_img:       np.ndarray,
    out_width:        int,
    use_poisson:      bool,
    smooth_tracking:  bool,
    match_brightness: bool,
    add_reflection:   bool,
) -> str:

    cap = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 24
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale   = out_width / src_w
    out_h   = int(src_h * scale)
    out_w   = out_width

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # ── UI elements ──────────────────────────────────────────────────────────
    progress_bar = st.progress(0.0)
    status_text  = st.empty()
    preview_slot = st.empty()

    # ── Tracking state ────────────────────────────────────────────────────────
    prev_corners: np.ndarray | None = None
    prev_gray:    np.ndarray | None = None
    kalman:       CornerKalman | None = None
    detect_interval = 8     # re-detect every N frames (handles large motion)
    frame_idx  = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = frame.copy()

        # ── Corner detection / tracking ──────────────────────────────────────
        if prev_corners is None or (frame_idx % detect_interval == 0):
            # Full GrabCut re-detection
            corners = detect_phone_corners(frame)
            if corners is not None:
                if smooth_tracking and kalman is None:
                    kalman = CornerKalman(corners)
                prev_corners = corners
            # If detection fails and we have old corners, keep them
        else:
            # Fast optical-flow tracking between consecutive frames
            tracked = track_corners_optflow(prev_gray, gray, prev_corners)
            if tracked is not None:
                prev_corners = tracked
            else:
                # Tracking lost — force re-detect next frame
                prev_corners = None

        # ── Kalman smoothing ─────────────────────────────────────────────────
        if smooth_tracking and kalman is not None and prev_corners is not None:
            prev_corners = kalman.update(prev_corners)

        # ── Composite ────────────────────────────────────────────────────────
        if prev_corners is not None:
            output_frame = composite_iphone(
                frame, iphone_img, prev_corners,
                use_poisson=use_poisson,
                match_brightness=match_brightness,
                add_reflection=add_reflection,
            )

        writer.write(output_frame)

        # ── UI updates ────────────────────────────────────────────────────────
        pct = (frame_idx + 1) / max(total_frames, 1)
        progress_bar.progress(min(pct, 1.0))

        elapsed = time.time() - t0
        fps_proc = (frame_idx + 1) / max(elapsed, 0.001)
        eta      = (total_frames - frame_idx - 1) / max(fps_proc, 0.001)
        status_text.markdown(
            f"⚙️ Frame **{frame_idx+1}/{total_frames}** &nbsp;|&nbsp; "
            f"{fps_proc:.1f} fps &nbsp;|&nbsp; ETA **{eta:.0f}s**"
        )

        # Live preview every 15 frames
        if frame_idx % 15 == 0:
            preview_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            preview_slot.image(preview_rgb, caption="Live preview", use_container_width=True)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
if video_file and zip_file:
    st.markdown("---")

    # Preview uploaded video
    st.subheader("📹 Input video")
    st.video(video_file)

    if st.button("🚀 Replace Phone with iPhone 14 Pro", type="primary"):

        # ── Save video to temp file ──────────────────────────────────────────
        with st.spinner("📦 Extracting iPhone texture from ZIP…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
                tmp_v.write(video_file.read())
                video_path = tmp_v.name

            iphone_img = extract_iphone_texture(zip_file.read())

        if iphone_img is None:
            st.error("❌ Could not extract iPhone texture from ZIP. "
                     "Make sure the ZIP contains a PNG/JPG baseColor texture.")
            st.stop()

        h_i, w_i = iphone_img.shape[:2]
        st.success(f"✅ iPhone 14 Pro front-face texture loaded  ({w_i}×{h_i} px)")

        col_a, col_b = st.columns(2)
        with col_a:
            iphone_preview = cv2.cvtColor(iphone_img, cv2.COLOR_BGR2RGB)
            st.image(iphone_preview, caption="iPhone 14 Pro texture (front face)",
                     use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ Processing…")

        # ── Run pipeline ─────────────────────────────────────────────────────
        out_path = process_video(
            video_path       = video_path,
            iphone_img       = iphone_img,
            out_width        = output_w,
            use_poisson      = use_poisson,
            smooth_tracking  = smooth_tracking,
            match_brightness = match_brightness,
            add_reflection   = add_reflection,
        )

        st.success("✅ Done!")
        st.markdown("---")
        st.subheader("🎬 Output video")
        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download output_replaced.mp4",
                data=f.read(),
                file_name="output_replaced.mp4",
                mime="video/mp4",
            )

elif not video_file and not zip_file:
    st.info("👆 Upload both the video and the iPhone 14 Pro ZIP to get started.")
elif not video_file:
    st.warning("📹 Please upload the input video.")
elif not zip_file:
    st.warning("📦 Please upload the iPhone 14 Pro ZIP file.")
