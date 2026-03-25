"""
Ultimate Phone Replacement System
================================
Supports:
✅ 2D Image
✅ 3D Model (.glb / .obj)
✅ Texture ZIP
"""

import subprocess, sys, importlib
import streamlit as st
import cv2, numpy as np, tempfile, os, zipfile

# ---------------- INSTALL ----------------
def install(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

install("opencv-python", "cv2")
install("numpy")
install("streamlit")
install("ultralytics")
install("trimesh")
install("pyrender")

from ultralytics import YOLO
import trimesh
import pyrender

# ---------------- UI ----------------
st.set_page_config(page_title="Ultimate Phone Replacement")
st.title("📱 Ultimate Phone Replacement System")

video_file = st.file_uploader("📹 Upload Video", type=["mp4","avi","mov"])

mode = st.radio("Select Input Type", ["2D Image", "3D Model", "Texture ZIP"])

if mode == "2D Image":
    input_file = st.file_uploader("Upload Phone Image", type=["png","jpg","jpeg"])

elif mode == "3D Model":
    input_file = st.file_uploader("Upload 3D Model", type=["glb","obj"])

else:
    input_file = st.file_uploader("Upload Texture ZIP", type=["zip"])

output_w = st.selectbox("Output width", [540,720,1080])

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8m-seg.pt")

model = load_model()

# ---------------- TEXTURE LOADER ----------------
def load_texture_from_zip(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "file.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith((".png",".jpg",".jpeg")):
                    return cv2.imread(os.path.join(root, file))

    return None

# ---------------- 3D RENDER ----------------
def render_3d_model(model_path, size=256):
    try:
        mesh = trimesh.load(model_path)

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
        scene.add(camera, pose=np.eye(4))

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))

        r = pyrender.OffscreenRenderer(size, size)
        color, _ = r.render(scene)

        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    except:
        return None

# ---------------- CORE REPLACEMENT ----------------
def replace_phone(frame, mask, phone_img):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    c = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    w, h = int(rect[1][0]), int(rect[1][1])

    if w < 20 or h < 20:
        return frame

    phone_resized = cv2.resize(phone_img, (w, h))

    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = box.astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(phone_resized, M, (frame.shape[1], frame.shape[0]))

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(warped, warped, mask=mask)

    return cv2.add(bg, fg)

# ---------------- PROCESS VIDEO ----------------
def process(video_path, phone_img, out_w):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ow = int(cap.get(3))
    oh = int(cap.get(4))

    scale = out_w / ow
    out_h = int(oh * scale)

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prog = st.progress(0)

    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (out_w, out_h))

        res = model(frame, verbose=False)[0]

        mask = np.zeros((out_h, out_w), dtype=np.uint8)

        if res.masks is not None:
            for m, cls in zip(res.masks.data.cpu().numpy(),
                              res.boxes.cls.cpu().numpy()):

                if model.names[int(cls)].lower() != "cell phone":
                    continue

                bin_mask = (m > 0.5).astype(np.uint8) * 255

# 🔥 FIX: resize mask to frame size
                bin_mask = cv2.resize(bin_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

                mask = cv2.bitwise_or(mask, bin_mask)

        # clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        frame = replace_phone(frame, mask, phone_img)

        writer.write(frame)

        i += 1
        prog.progress(min(i/total,1.0))

    cap.release()
    writer.release()

    return out_path

# ---------------- RUN ----------------
if video_file and input_file:

    st.video(video_file)

    # save video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    # -------- LOAD INPUT --------
    if mode == "2D Image":
        file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
        phone_img = cv2.imdecode(file_bytes, 1)

    elif mode == "3D Model":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp:
            tmp.write(input_file.read())
            model_path = tmp.name

        phone_img = render_3d_model(model_path)

        if phone_img is None:
            st.error("❌ 3D render failed")
            st.stop()

    else:  # Texture ZIP
        phone_img = load_texture_from_zip(input_file)

        if phone_img is None:
            st.error("❌ No texture found in ZIP")
            st.stop()

    # -------- PROCESS --------
    if st.button("▶️ Run Replacement"):

        with st.spinner("Processing..."):
            out = process(video_path, phone_img, output_w)

        st.success("✅ Done")
        st.video(out)

        with open(out, "rb") as f:
            st.download_button("Download Result", f.read(), "output.mp4")
