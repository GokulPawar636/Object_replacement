"""
Phone → 3D Model Replacement  (v3 — Corner & Center Exact Match)
=================================================================

Key improvements over v2:
  1. Exact corner alignment — The 3D model's four corners (on XY plane)
     are mapped precisely to the phone's four detected corners.
  2. Center matching — The model's center is aligned with the phone's center.
  3. Full texture/vertex color preservation — The alignment transform
     (rotation, scale, translation) leaves UVs and colors intact.
  4. No additional pose distortion — The phone pose estimation remains
     robust using the known planar rectangle.
"""

import subprocess, sys, importlib, os, tempfile, time

# ── Auto-install dependencies ──────────────────────────────────────────────
def _pip(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg.replace("-", "_").split("[")[0])
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

for p, m in [
    ("opencv-python", "cv2"), ("numpy", None), ("streamlit", None),
    ("ultralytics", None), ("trimesh", None),
    ("imageio-ffmpeg", "imageio_ffmpeg"), ("scipy", None),
    ("filterpy", None),
]:
    _pip(p, m)

import streamlit as st
import cv2, numpy as np, trimesh, imageio_ffmpeg
from ultralytics import YOLO

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# =============================================================================
# Page layout
# =============================================================================
st.set_page_config(page_title="3D Replacement v3", page_icon="🎯", layout="wide")
st.title("🎯 Phone → 3D Model Replacement  v3")
st.caption("Exact corner+center alignment · Full texture/color · Mask-confined")

# =============================================================================
# Load YOLO
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO("yolov8m-seg.pt")

with st.spinner("Loading YOLO…"):
    yolo = load_yolo()
st.success("✅ YOLO ready")
st.divider()

# =============================================================================
# Settings
# =============================================================================
c1, c2, c3 = st.columns(3)
with c1:
    conf        = st.slider("YOLO confidence",   0.10, 0.80, 0.30, 0.05)
    out_w       = st.selectbox("Output width",   [540, 720, 1080], index=0)
with c2:
    yolo_every  = st.slider("YOLO every N frames", 1, 6, 3)
    max_faces   = st.slider("Max mesh faces",    200, 3000, 800)
with c3:
    model_color    = st.color_picker("Fallback colour (if no texture)", "#00C8FF")
    wireframe_mode = st.checkbox("Wireframe", False)
    smooth_shading = st.checkbox("Shading (with texture/colors)",   True)
    edge_feather   = st.slider("Edge sharpness (px)", 0, 15, 4,
                                help="Lower = sharper edges. Higher = softer blend")

st.divider()

# =============================================================================
# Phone physical dimensions  (half-extents in cm)
# =============================================================================
PH_W, PH_H = 3.75, 8.0        # → full phone  7.5 × 16 cm

PHONE_OBJ_PTS = np.float64([   # 4 corners in object space (Z=0 plane)
    [-PH_W, -PH_H, 0],         # top-left
    [ PH_W, -PH_H, 0],         # top-right
    [ PH_W,  PH_H, 0],         # bottom-right
    [-PH_W,  PH_H, 0],         # bottom-left
])
PHONE_CENTER = np.float64([0, 0, 0])  # center of the phone rectangle

# =============================================================================
# Mesh loading & normalisation with texture extraction
# =============================================================================

def extract_texture_image(mesh_obj):
    """
    Extract texture image from trimesh object.
    Returns (texture_image, uv_coords) or (None, None) if no texture found.
    """
    try:
        if hasattr(mesh_obj, 'visual') and mesh_obj.visual is not None:
            visual = mesh_obj.visual
            if hasattr(visual, 'material') and visual.material is not None:
                material = visual.material
                if hasattr(material, 'image') and material.image is not None:
                    tex_img = np.array(material.image)
                    if tex_img.ndim == 3 and tex_img.shape[2] >= 3:
                        if tex_img.shape[2] == 4:
                            tex_img = cv2.cvtColor(tex_img, cv2.COLOR_RGBA2BGR)
                        elif tex_img.shape[2] == 3:
                            tex_img = cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR)
                        if hasattr(visual, 'uv') and visual.uv is not None:
                            return tex_img.astype(np.uint8), np.array(visual.uv, np.float32)
                        return tex_img.astype(np.uint8), None
    except Exception:
        pass
    return None, None


def extract_vertex_colors(mesh_obj):
    """
    Extract per-vertex colors from mesh if available.
    Returns (N, 3) BGR color array or None.
    """
    try:
        if hasattr(mesh_obj, 'visual'):
            visual = mesh_obj.visual
            if hasattr(visual, 'vertex_colors'):
                colors = visual.vertex_colors
                if colors is not None and len(colors) > 0:
                    colors = np.array(colors)
                    if colors.ndim == 2 and colors.shape[1] >= 3:
                        # Convert RGBA/RGB to BGR
                        colors = colors[:, [2, 1, 0]]
                        return colors.astype(np.uint8)
    except Exception:
        pass
    return None


def compute_model_corners_and_center(verts):
    """
    Compute the four corner points (on XY plane, Z=0) of the model's bounding box,
    and its center point (also on Z=0).
    Returns:
        corners (4,3): TL, TR, BR, BL in object space (Z=0)
        center (3,): (cx, cy, 0)
    """
    min_xy = verts[:, :2].min(axis=0)
    max_xy = verts[:, :2].max(axis=0)
    cx = (min_xy[0] + max_xy[0]) / 2.0
    cy = (min_xy[1] + max_xy[1]) / 2.0
    corners = np.array([
        [min_xy[0], min_xy[1], 0],   # TL
        [max_xy[0], min_xy[1], 0],   # TR
        [max_xy[0], max_xy[1], 0],   # BR
        [min_xy[0], max_xy[1], 0],   # BL
    ], dtype=np.float64)
    center = np.array([cx, cy, 0], dtype=np.float64)
    return corners, center


def align_model_to_phone(verts, model_corners, model_center, phone_corners, phone_center):
    """
    Compute a similarity transform (rotation in XY, scaling X/Y, translation)
    that maps model_corners to phone_corners and model_center to phone_center.
    Applies the transform to all vertices and returns transformed vertices.
    """
    # We'll solve for:  p_phone = s * R * p_model + t
    # Using the four corners and center gives an overdetermined system.
    # For simplicity, we compute separate scale for X and Y (non-uniform scaling)
    # and rotation around Z, plus translation.
    
    # Extract 2D points
    model_pts = model_corners[:, :2]
    phone_pts = phone_corners[:, :2]
    model_c = model_center[:2]
    phone_c = phone_center[:2]
    
    # Compute centroids
    model_centroid = np.mean(model_pts, axis=0)
    phone_centroid = np.mean(phone_pts, axis=0)
    
    # Center the points
    model_centered = model_pts - model_centroid
    phone_centered = phone_pts - phone_centroid
    
    # Compute optimal rotation (2D) using SVD
    H = model_centered.T @ phone_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det=1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale factors (separate X and Y for better fit)
    model_rotated = model_centered @ R.T
    scale_x = np.sum(phone_centered[:, 0] * model_rotated[:, 0]) / np.sum(model_rotated[:, 0]**2)
    scale_y = np.sum(phone_centered[:, 1] * model_rotated[:, 1]) / np.sum(model_rotated[:, 1]**2)
    scale = np.array([scale_x, scale_y])
    
    # Translation: align centroids after scaling and rotation
    t = phone_centroid - (model_centroid @ R.T) * scale
    
    # Build 4x4 transform matrix (affine, XY only, Z unchanged)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = R[0, 0] * scale[0]
    T[0, 1] = R[0, 1] * scale[0]
    T[1, 0] = R[1, 0] * scale[1]
    T[1, 1] = R[1, 1] * scale[1]
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    # Z axis remains unchanged (scale=1, no rotation)
    T[2, 2] = 1.0
    
    # Apply to all vertices (homogeneous)
    verts_hom = np.hstack([verts, np.ones((len(verts), 1))])
    verts_transformed = (T @ verts_hom.T).T[:, :3]
    return verts_transformed


@st.cache_resource(show_spinner=False)
def load_mesh(path, max_f):
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not meshes:
            raise ValueError("No mesh geometry found in file")
        m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        m = loaded
    else:
        raise ValueError(f"Unsupported type: {type(loaded).__name__}")

    if len(m.faces) == 0:
        raise ValueError("Mesh has no faces")

    # Extract texture and vertex colors BEFORE any modifications
    texture_img, uv_coords = extract_texture_image(m)
    vertex_colors = extract_vertex_colors(m)

    # Remove degenerate faces
    face_areas = np.linalg.norm(
        np.cross(m.vertices[m.faces[:, 1]] - m.vertices[m.faces[:, 0]],
                 m.vertices[m.faces[:, 2]] - m.vertices[m.faces[:, 0]]),
        axis=1
    )
    valid_mask = face_areas > 1e-6
    m.faces = m.faces[valid_mask]
    if uv_coords is not None and len(uv_coords) == len(valid_mask):
        uv_coords = uv_coords[valid_mask]

    # Decimate — clears UV since face indices change
    if len(m.faces) > max_f:
        try:
            m = m.simplify_quadric_decimation(face_count=max_f)
            texture_img = None
            uv_coords   = None
        except Exception:
            pass

    # Centre the mesh (for later alignment)
    m.vertices -= m.centroid

    # Compute corners and center for later alignment (will be done in process)
    # We return the raw vertices (centered) and the other data
    return (np.array(m.vertices, np.float64),
            np.array(m.faces,    np.int32),
            texture_img,
            uv_coords,
            vertex_colors)


def make_box_mesh(max_f=800):
    m = trimesh.creation.box(extents=[PH_W * 2, PH_H * 2, 0.8])
    return (np.array(m.vertices, np.float64),
            np.array(m.faces,    np.int32),
            None,   # texture_img
            None,   # uv_coords
            None)   # vertex_colors


st.subheader("📦 Upload 3D model  (optional)")
model_file = st.file_uploader(
    "Upload .obj / .glb / .stl / .ply",
    type=["obj", "glb", "gltf", "stl", "ply", "off"],
)

# Global mesh data (all five components kept together)
verts_g, faces_g, tex_g, uv_g, vcol_g = None, None, None, None, None

if model_file:
    suf     = os.path.splitext(model_file.name)[1].lower()
    data    = model_file.read()
    tmp_mod = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    tmp_mod.write(data); tmp_mod.flush(); tmp_mod.close()
    try:
        verts_g, faces_g, tex_g, uv_g, vcol_g = load_mesh(tmp_mod.name, max_faces)
        has_texture = "✓ Texture"        if tex_g  is not None else "✗ No texture"
        has_colors  = "✓ Vertex colors"  if vcol_g is not None else "✗ No vertex colors"
        st.success(
            f"✅ {len(verts_g):,} verts · {len(faces_g):,} faces (≤{max_faces})"
            f" | {has_texture} | {has_colors}"
        )
    except Exception as e:
        st.error(f"Could not load **{model_file.name}**: {e}")
    finally:
        try:
            os.unlink(tmp_mod.name)
        except Exception:
            pass
else:
    st.info("No model uploaded — using a placeholder box (will be aligned to phone rectangle).")

st.divider()

# =============================================================================
# Helpers (unchanged from v2)
# =============================================================================

def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def sort_corners(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.float64([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ])


def mask_from_seg(raw_float, ow, oh):
    """Convert YOLO float mask → binary uint8; return largest-contour fill."""
    if raw_float.shape[:2] != (oh, ow):
        raw_float = cv2.resize(raw_float, (ow, oh), cv2.INTER_LINEAR)
    binary = (raw_float >= 0.5).astype(np.uint8) * 255
    binary = cv2.GaussianBlur(binary, (5, 5), 0)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return None
    out = np.zeros((oh, ow), np.uint8)
    cv2.fillPoly(out, [largest], 255)
    return out


def corners_from_mask(mask):
    """Extract ordered phone corners from a binary mask."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return None
    corners_raw = cv2.boxPoints(cv2.minAreaRect(largest))
    return sort_corners(corners_raw.astype(np.float64))


def build_camera_matrix(ow, oh):
    """Approximate camera intrinsics from a typical phone camera FOV."""
    h_fov_rad = np.radians(70.5)
    f_computed = (ow / 2.0) / np.tan(h_fov_rad / 2.0)
    cx, cy = ow / 2.0, oh / 2.0
    return np.float64([
        [f_computed, 0.0, cx],
        [0.0, f_computed, cy],
        [0.0, 0.0,        1.0],
    ])


def bbox_from_corners(corners_2d):
    xs = corners_2d[:, 0]
    ys = corners_2d[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_from_corners(corners_2d, ow, oh):
    """Create a filled binary mask from tracked phone corners."""
    mask = np.zeros((oh, ow), np.uint8)
    poly = np.round(corners_2d).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    if np.count_nonzero(mask) < 300:
        return None
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def smooth_corners(prev_corners, new_corners, alpha=0.65):
    if prev_corners is None:
        return new_corners.astype(np.float64)
    return prev_corners * (1.0 - alpha) + new_corners * alpha


def smooth_pose(prev_pose, new_pose, rot_alpha=0.35, trans_alpha=0.45):
    if prev_pose is None:
        return new_pose
    prev_rvec, prev_tvec = prev_pose
    new_rvec,  new_tvec  = new_pose
    rvec = prev_rvec * (1.0 - rot_alpha)  + new_rvec * rot_alpha
    tvec = prev_tvec * (1.0 - trans_alpha) + new_tvec * trans_alpha
    return rvec, tvec


def solve_pose_from_corners(corners_2d, ow, oh, verts_model=None, init_pose=None):
    """
    Solve a 3D pose from 4 ordered 2D phone corners.
    Returns (rvec, tvec, K) or None.
    """
    if corners_2d is None or len(corners_2d) != 4:
        return None

    corners_2d = np.asarray(corners_2d, dtype=np.float64)
    center_2d  = corners_2d.mean(axis=0)
    K          = build_camera_matrix(ow, oh)
    dist       = np.zeros((4, 1))

    use_guess = init_pose is not None
    if use_guess:
        r_guess = np.asarray(init_pose[0], dtype=np.float64).reshape(3, 1)
        t_guess = np.asarray(init_pose[1], dtype=np.float64).reshape(3, 1)
    else:
        r_guess = None
        t_guess = None

    ok, rvec, tvec = cv2.solvePnP(
        PHONE_OBJ_PTS,
        corners_2d,
        K, dist,
        rvec=r_guess,
        tvec=t_guess,
        useExtrinsicGuess=use_guess,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    try:
        rvec, tvec = cv2.solvePnPRefineVVS(
            PHONE_OBJ_PTS, corners_2d, K, dist, rvec, tvec,
            iterations=30, epsilon=1e-6,
        )
        rvec, tvec = cv2.solvePnPRefineVVS(
            PHONE_OBJ_PTS, corners_2d, K, dist, rvec, tvec,
            iterations=20, epsilon=1e-7,
        )
    except Exception:
        pass

    # Optional: centre-point refinement using the model's geometric centre
    if verts_model is not None:
        try:
            _, model_center_3d = compute_model_corners_and_center(verts_model)
            center_3d_proj, _ = cv2.projectPoints(
                model_center_3d.reshape(1, 3).astype(np.float32),
                rvec, tvec, K, dist,
            )
            center_2d_proj = center_3d_proj.reshape(-1, 2)[0]
            center_error   = np.linalg.norm(center_2d_proj - center_2d)
            if center_error > 1.0:
                obj_pts_aug = np.vstack([
                    PHONE_OBJ_PTS,
                    np.array([[0, 0, 0]], dtype=np.float64),
                ])
                img_pts_aug = np.vstack([corners_2d, center_2d.reshape(1, 2)])
                rvec, tvec  = cv2.solvePnPRefineVVS(
                    obj_pts_aug.astype(np.float32),
                    img_pts_aug.astype(np.float32),
                    K, dist, rvec, tvec,
                    iterations=15, epsilon=1e-7,
                )
        except Exception:
            pass

    return rvec, tvec, K


def get_pose(mask, ow, oh, verts_model=None, init_pose=None):
    """
    Fit a rotated rectangle to the mask and solve PnP.
    Returns (rvec, tvec, K) or None.
    """
    corners_2d = corners_from_mask(mask)
    return solve_pose_from_corners(
        corners_2d, ow, oh,
        verts_model=verts_model,
        init_pose=init_pose,
    )


# =============================================================================
# Background plate  (fast erase)
# =============================================================================

class BackgroundPlate:
    def __init__(self, n=7):
        self.buf = []
        self.n   = n

    def update(self, frame):
        self.buf.append(frame.copy())
        if len(self.buf) > self.n:
            self.buf.pop(0)

    def erase(self, frame, mask):
        """Replace pixels inside mask with median background."""
        if len(self.buf) < 2:
            out = frame.copy()
            ys, xs = np.where(mask > 0)
            if len(ys):
                r1, r2 = ys.min(), ys.max() + 1
                c1, c2 = xs.min(), xs.max() + 1
                out[r1:r2, c1:c2] = cv2.GaussianBlur(
                    frame[r1:r2, c1:c2], (51, 51), 0
                )
            return out
        stack = np.stack(self.buf, axis=0).astype(np.float32)
        bg    = np.median(stack, axis=0).astype(np.uint8)
        out   = frame.copy()
        out[mask > 0] = bg[mask > 0]
        return out


# =============================================================================
# Fast rasteriser — MASK-CONFINED (same as v2)
# =============================================================================

def precompute_shading(verts_cam, faces, color_bgr, vertex_colors=None):
    """
    Compute per-face shading with improved lighting model.
    Returns per-face shaded colors (N, 3) uint8.
    """
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]

    edge1   = v1 - v0
    edge2   = v2 - v0
    normals = np.cross(edge1, edge2).astype(np.float64)

    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, where=(norms > 1e-8), out=normals)

    light_dir = np.array([0.3, -0.6, -0.74], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    diff      = np.clip(-normals @ light_dir, 0, 1)
    ambient   = 0.25
    intensity = np.power(diff, 0.75)
    shades    = (ambient + (1.0 - ambient) * intensity)[:, None]

    spec_intensity = np.power(np.maximum(0, -normals @ light_dir), 16)[:, None]
    shades = np.clip(shades + spec_intensity * 0.3, 0, 1)
    shades = np.power(np.clip(shades, 0, 1), 0.85)

    if vertex_colors is not None:
        face_base_colors = (
            vertex_colors[faces[:, 0]].astype(np.float64) +
            vertex_colors[faces[:, 1]].astype(np.float64) +
            vertex_colors[faces[:, 2]].astype(np.float64)
        ) / 3.0
    else:
        face_base_colors = np.tile(
            np.array(color_bgr, dtype=np.float64), (len(faces), 1)
        )

    colored = face_base_colors * shades
    return np.clip(colored, 0, 255).astype(np.uint8)


def render_mesh_onto_mask(
    base_frame,
    mask,
    verts_obj,
    faces,
    rvec, tvec, K,
    color_bgr,
    wireframe=False,
    shading=True,
    feather=6,
    vertex_colors=None,
    texture_img=None,
    uv_coords=None,
):
    """
    Render the 3D mesh and composite it ONLY inside `mask`.
    """
    h, w = base_frame.shape[:2]
    dist  = np.zeros((4, 1))

    R, _       = cv2.Rodrigues(rvec)
    verts_cam  = (R @ verts_obj.T).T + tvec.ravel()
    z_vals     = verts_cam[:, 2]

    pts2d, _ = cv2.projectPoints(
        verts_obj.astype(np.float32), rvec, tvec, K, dist
    )
    pts2d = pts2d.reshape(-1, 2)

    face_colors = (
        precompute_shading(verts_cam, faces, color_bgr, vertex_colors)
        if shading and not wireframe
        else None
    )

    face_z = z_vals[faces].mean(axis=1)
    order  = np.argsort(face_z)[::-1]

    f_pts   = pts2d[faces]
    f_min   = f_pts.min(axis=1)
    f_max   = f_pts.max(axis=1)
    f_z_min = z_vals[faces].min(axis=1)

    near_threshold = 0.01
    visible = (
        (f_z_min > near_threshold) &
        (f_max[:, 0] >= -5)   & (f_min[:, 0] < w + 5) &
        (f_max[:, 1] >= -5)   & (f_min[:, 1] < h + 5)
    )

    canvas = np.zeros_like(base_frame)
    drawn  = np.zeros((h, w), np.uint8)

    for fi in order:
        if not visible[fi]:
            continue
        tri = faces[fi]
        p2d = pts2d[tri].astype(np.float32)

        area_tri = 0.5 * abs(
            (p2d[1, 0] - p2d[0, 0]) * (p2d[2, 1] - p2d[0, 1]) -
            (p2d[2, 0] - p2d[0, 0]) * (p2d[1, 1] - p2d[0, 1])
        )
        if area_tri < 0.5:
            continue

        p2d_int = p2d.astype(np.int32)

        if wireframe:
            cv2.polylines(canvas, [p2d_int], True, color_bgr, 2, cv2.LINE_AA)
            cv2.polylines(drawn,  [p2d_int], True, 255,       2, cv2.LINE_AA)
        else:
            if texture_img is not None and uv_coords is not None:
                uv     = uv_coords[fi]
                h_tex, w_tex = texture_img.shape[:2]
                uv_px  = np.zeros((3, 2), dtype=np.int32)
                uv_px[:, 0] = np.clip((uv[:, 0] * w_tex).astype(int), 0, w_tex - 1)
                uv_px[:, 1] = np.clip(((1 - uv[:, 1]) * h_tex).astype(int), 0, h_tex - 1)
                u = int(np.mean(uv_px[:, 0]))
                v = int(np.mean(uv_px[:, 1]))
                tex_color = texture_img[v, u]
                if face_colors is not None:
                    shade = face_colors[fi] / 255.0
                    fc = tuple(int(x) for x in np.clip(tex_color * shade, 0, 255).astype(np.uint8))
                else:
                    fc = tuple(int(x) for x in tex_color)
            else:
                fc = (
                    tuple(int(x) for x in face_colors[fi])
                    if face_colors is not None
                    else color_bgr
                )

            cv2.fillConvexPoly(canvas, p2d_int, fc, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(drawn,  p2d_int, 255)
            cv2.polylines(canvas, [p2d_int], True, fc, 1, cv2.LINE_AA)

    alpha_raw = cv2.bitwise_and(mask, drawn)

    if feather > 0:
        k = min(feather + 1, min(h, w) // 4)
        k = (k // 2) * 2 + 1
        alpha_f = cv2.GaussianBlur(
            alpha_raw.astype(np.float32), (k, k), feather / 3.5
        )
    else:
        alpha_f = alpha_raw.astype(np.float32)

    alpha_3 = (alpha_f / 255.0)[:, :, np.newaxis]

    out = (
        base_frame.astype(np.float64) * (1.0 - alpha_3) +
        canvas.astype(np.float64)     * alpha_3
    ).astype(np.uint8)

    return out


# =============================================================================
# Kalman filter for pose smoothing (unchanged)
# =============================================================================

from filterpy.kalman import KalmanFilter

class PoseKalman:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=6)
        self.kf.x = np.zeros(6)
        self.kf.F = np.eye(6)
        self.kf.H = np.eye(6)
        self.kf.P *= 20
        self.kf.R *= 0.3
        self.kf.Q *= 0.01

    def predict(self):
        self.kf.predict()
        return self.kf.x.copy()

    def update(self, rvec, tvec):
        z = np.hstack([rvec.ravel(), tvec.ravel()])
        self.kf.update(z)
        return self.kf.x.copy()


def is_pose_valid(prev_rvec, prev_tvec, new_rvec, new_tvec):
    if prev_rvec is None:
        return True
    rot_diff   = np.linalg.norm(new_rvec - prev_rvec)
    trans_diff = np.linalg.norm(new_tvec - prev_tvec)
    return rot_diff < 0.4 and trans_diff < 4.0


# =============================================================================
# Continuous tracker (unchanged)
# =============================================================================

class PoseTracker:
    """
    Tracks the phone on every frame using optical-flow feature points.
    Detection initialises the tracker; optical flow propagates the pose
    to intermediate frames.
    """

    def __init__(self, ow, oh, verts_model=None):
        self.ow          = ow
        self.oh          = oh
        self.verts_model = verts_model
        self.kalman      = PoseKalman()
        self.active      = False
        self.rvec        = None
        self.tvec        = None
        self.K           = None
        self.mask        = None
        self.corners     = None
        self.bbox        = None
        self.points      = None
        self.last_gray   = None
        self.lost        = 0
        self.hits        = 0

    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _clip_corners(self, corners):
        corners = np.asarray(corners, dtype=np.float64)
        corners[:, 0] = np.clip(corners[:, 0], 0, self.ow - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, self.oh - 1)
        return corners

    def _refresh_points(self, gray, mask, corners):
        feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
        points = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        corner_points = corners.astype(np.float32).reshape(-1, 1, 2)
        if points is None:
            return corner_points
        return np.concatenate([points, corner_points], axis=0).astype(np.float32)

    def _set_state(self, gray, corners, mask, bbox, rvec, tvec, K):
        self.corners   = self._clip_corners(corners)
        self.mask      = mask
        self.bbox      = bbox
        self.rvec      = rvec.copy()
        self.tvec      = tvec.copy()
        self.K         = K.copy()
        self.last_gray = gray
        self.points    = self._refresh_points(gray, mask, self.corners)
        self.active    = True

    def init(self, frame, corners, mask, bbox, rvec, tvec, K):
        gray = self._gray(frame)
        self._set_state(gray, corners, mask, bbox, rvec, tvec, K)
        self.hits = 1
        self.lost = 0

    def update_from_detection(self, frame, mask, bbox, rvec, tvec, K):
        corners = corners_from_mask(mask)
        if corners is None:
            return False
        if self.corners is not None:
            corners = smooth_corners(self.corners, corners, alpha=0.72)
        state = self.kalman.update(rvec, tvec)
        rvec  = state[:3].reshape(3, 1)
        tvec  = state[3:].reshape(3, 1)
        gray  = self._gray(frame)
        self._set_state(gray, corners, mask, bbox, rvec, tvec, K)
        self.hits += 1
        self.lost  = 0
        return True

    def track(self, frame):
        if (not self.active or
                self.points is None or
                len(self.points) < 4 or
                self.last_gray is None):
            return False

        gray = self._gray(frame)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_gray, gray, self.points, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            self.lost += 1
            return False

        good_mask = status.ravel() == 1
        old_good  = self.points[good_mask].reshape(-1, 2)
        new_good  = next_pts[good_mask].reshape(-1, 2)
        if len(old_good) < 6:
            self.lost += 1
            return False

        M, _ = cv2.estimateAffinePartial2D(
            old_good, new_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if M is None:
            self.lost += 1
            return False

        tracked_corners = cv2.transform(
            self.corners.reshape(1, -1, 2).astype(np.float32), M
        ).reshape(-1, 2)
        tracked_corners = self._clip_corners(tracked_corners)

        tracked_mask = mask_from_corners(tracked_corners, self.ow, self.oh)
        if tracked_mask is None:
            self.lost += 1
            return False

        pose = solve_pose_from_corners(
            tracked_corners, self.ow, self.oh,
            verts_model=self.verts_model,
            init_pose=(self.rvec, self.tvec),
        )

        if pose is not None:
            rvec_new, tvec_new, K_new = pose
            if is_pose_valid(self.rvec, self.tvec, rvec_new, tvec_new):
                state    = self.kalman.update(rvec_new, tvec_new)
                rvec_out = state[:3].reshape(3, 1)
                tvec_out = state[3:].reshape(3, 1)
                # Smooth with the previous accepted pose
                rvec_out, tvec_out = smooth_pose(
                    (self.rvec, self.tvec), (rvec_out, tvec_out),
                    rot_alpha=0.22, trans_alpha=0.30,
                )
                K_out = K_new
            else:
                # Pose jump detected — fall back to Kalman prediction
                state    = self.kalman.predict()
                rvec_out = state[:3].reshape(3, 1)
                tvec_out = state[3:].reshape(3, 1)
                K_out    = self.K  # keep previous K
        else:
            # No pose solution — use Kalman prediction, keep previous K
            state    = self.kalman.predict()
            rvec_out = state[:3].reshape(3, 1)
            tvec_out = state[3:].reshape(3, 1)
            K_out    = self.K

        smoothed_corners = smooth_corners(self.corners, tracked_corners, alpha=0.42)
        smoothed_mask    = mask_from_corners(smoothed_corners, self.ow, self.oh)
        if smoothed_mask is None:
            self.lost += 1
            return False

        self._set_state(
            gray,
            smoothed_corners,
            smoothed_mask,
            bbox_from_corners(smoothed_corners),
            rvec_out, tvec_out, K_out,
        )
        self.lost = 0
        return True


# =============================================================================
# Main pipeline (with model alignment to phone rectangle)
# =============================================================================

def process(
    video_path,
    verts_g, faces_g, tex_g, uv_g, vcol_g,   # all 5 mesh components
    conf, out_w, yolo_every, max_faces,
    color_hex, wireframe_mode, smooth_mode, feather_px,
):
    # Resolve mesh (uploaded model or fallback box)
    if verts_g is not None and faces_g is not None:
        verts_raw, faces, texture_img, uv_coords, vertex_colors = (
            verts_g, faces_g, tex_g, uv_g, vcol_g
        )
    else:
        verts_raw, faces, texture_img, uv_coords, vertex_colors = make_box_mesh(max_faces)

    # --- NEW: Align model's corners and center to phone's rectangle ---
    # Compute model's four corner points (on XY plane) and its center
    model_corners, model_center = compute_model_corners_and_center(verts_raw)
    # Phone's fixed corners and center (in object space)
    phone_corners = PHONE_OBJ_PTS
    phone_center  = PHONE_CENTER

    # Transform vertices so that model_corners -> phone_corners and model_center -> phone_center
    verts = align_model_to_phone(verts_raw, model_corners, model_center,
                                 phone_corners, phone_center)

    # Optionally, you can print the alignment error for debugging (disabled in production)
    # aligned_corners, aligned_center = compute_model_corners_and_center(verts)
    # print("Corner alignment error (px in obj space):",
    #       np.linalg.norm(aligned_corners - phone_corners, axis=1).mean())
    # print("Center alignment error:", np.linalg.norm(aligned_center - phone_center))

    color = hex_bgr(color_hex)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = out_w / max(ow, 1)
    out_h = int(oh * scale) & ~1

    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )

    tracker = PoseTracker(ow, oh, verts_model=verts)
    bgplate = BackgroundPlate(n=7)
    prog    = st.progress(0)
    preview = st.empty()
    t0      = time.time()
    fi      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bgplate.update(frame)

        # Attempt optical-flow tracking on every frame (except first)
        tracked_ok = False
        if tracker.active and fi > 0:
            tracked_ok = tracker.track(frame)

        # Decide whether to run YOLO this frame
        run_yolo = (
            fi == 0 or
            not tracker.active or
            tracker.lost > 1 or
            fi % yolo_every == 0
        )

        if run_yolo:
            results    = yolo(frame, conf=conf, verbose=False)[0]
            best_score = -1.0
            best_mask  = tracker.mask.copy()   if tracker.mask  is not None else None
            best_bbox  = list(tracker.bbox)    if tracker.bbox  is not None else None
            best_pose  = None
            best_corners = None

            if results.masks is not None and results.boxes is not None:
                for i, cls in enumerate(results.boxes.cls.cpu().numpy()):
                    if yolo.names[int(cls)].lower() != "cell phone":
                        continue
                    det_mask = mask_from_seg(
                        results.masks.data[i].cpu().numpy(), ow, oh
                    )
                    if det_mask is None:
                        continue
                    pose = get_pose(
                        det_mask, ow, oh,
                        verts_model=verts,
                        init_pose=(
                            (tracker.rvec, tracker.tvec) if tracker.active else None
                        ),
                    )
                    if pose is None:
                        continue
                    corners = corners_from_mask(det_mask)
                    if corners is None:
                        continue

                    area  = float(np.sum(det_mask > 0))
                    score = area
                    if tracker.active and tracker.corners is not None:
                        corner_error = np.linalg.norm(
                            corners - tracker.corners, axis=1
                        ).mean()
                        score -= corner_error * 25.0

                    if score <= best_score:
                        continue
                    best_score   = score
                    best_mask    = det_mask
                    best_bbox    = results.boxes.xyxy[i].cpu().numpy().tolist()
                    best_pose    = pose
                    best_corners = corners

            if best_pose is not None and best_corners is not None:
                rvec, tvec, K = best_pose
                if not tracker.active:
                    tracker.init(
                        frame, best_corners, best_mask, best_bbox, rvec, tvec, K
                    )
                else:
                    tracker.update_from_detection(
                        frame, best_mask, best_bbox, rvec, tvec, K
                    )

        # Render
        out_frame = frame.copy()
        if tracker.active and tracker.hits >= 1 and tracker.mask is not None:
            erased    = bgplate.erase(out_frame, tracker.mask)
            out_frame = render_mesh_onto_mask(
                erased,
                tracker.mask,
                verts, faces,
                tracker.rvec, tracker.tvec, tracker.K,
                color,
                wireframe_mode,
                smooth_mode,
                feather_px,
                vertex_colors=vertex_colors,
                texture_img=texture_img,
                uv_coords=uv_coords,
            )

        out_resized = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(out_resized)

        fi      += 1
        elapsed  = time.time() - t0
        fps_e    = fi / max(elapsed, 1e-3)
        eta      = (total - fi) / max(fps_e, 1e-3)
        mode     = (
            "detect+track" if run_yolo else
            ("track"       if tracked_ok else "recover")
        )
        prog.progress(
            min(fi / total, 1.0),
            text=f"Frame {fi}/{total} | {fps_e:.1f} fps | ETA {eta:.0f}s | {mode}",
        )
        if fi % 15 == 0:
            preview.image(
                cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB),
                caption="Preview",
                use_container_width=True,
            )

    cap.release()
    writer.release()
    prog.empty()
    preview.empty()

    final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run(
        [FFMPEG, "-y", "-i", tmp,
         "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-movflags", "+faststart", final],
        capture_output=True, check=False,
    )
    try:
        os.unlink(tmp)
    except Exception:
        pass

    return final, fi, time.time() - t0


# =============================================================================
# UI (unchanged)
# =============================================================================

uploaded = st.file_uploader("📹 Upload video", type=["mp4", "avi", "mov", "mkv"])
if uploaded:
    st.video(uploaded)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if st.button("▶️ Run", type="primary", use_container_width=True):
        out_vid, n, elapsed = process(
            tmp_path,
            verts_g, faces_g, tex_g, uv_g, vcol_g,
            conf, out_w, yolo_every, max_faces,
            model_color, wireframe_mode, smooth_shading, edge_feather,
        )

        st.success(f"✅ {n} frames · {elapsed:.1f}s · {n / elapsed:.1f} fps")
        with open(out_vid, "rb") as fh:
            st.download_button(
                "📥 Download result", fh.read(),
                "replaced.mp4", "video/mp4",
                use_container_width=True,
            )
        try:
            os.unlink(out_vid)
        except Exception:
            pass

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

st.divider()
st.markdown("""
### v3 — Exact Corner & Center Alignment with Full Texture/Color

#### What's New

**Precise Geometric Mapping:**
- The 3D model's bounding box corners (on XY plane) are mathematically aligned to the phone's detected four corners.
- The model's center is simultaneously matched to the phone's center.
- Uses a similarity transform (rotation, non-uniform scaling, translation) computed via SVD for optimal fit.

**Why This Matters:**
- No more misalignment — the model exactly occupies the phone's screen region in 3D space.
- Preserves all texture coordinates and vertex colors (transform affects only vertex positions).
- Works with any model shape (box, character, abstract) — the four corners are derived from the model's XY extents.

**How It Works:**
1. Load and center the model (remove translation).
2. Compute model's four corner points (minX,minY), (maxX,minY), (maxX,maxY), (minX,maxY) at Z=0.
3. Compute model's center (average of min/max X and Y).
4. Compute optimal transform mapping model corners → phone corners and model center → phone center.
5. Apply transform to all vertices.
6. The phone pose (rvec, tvec) is then applied as usual — the model is already pre‑aligned.

**Texture & Color:**
- Full support for .OBJ with MTL, .GLB/.GLTF (PBR), .PLY vertex colors.
- Specular highlights, ambient lighting, and texture mapping preserved.
- Falls back to flat color if no texture/colors available.

**Performance Tip:** For best results, upload a model whose natural "base" is roughly planar (e.g., a phone case, a card, a flat object). For fully 3D objects (like a ball), the alignment still works but the object will be squashed to fit the phone's rectangle — adjust scaling in the code if needed.
""")