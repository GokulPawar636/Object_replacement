import os
import cv2
import json
import numpy as np
import math
import trimesh
import pyrender
import imageio

# =========================
# CONFIG
# =========================
FRAME_FOLDER = "frames"
POSE_FOLDER = "pose_output1"
OUTPUT_VIDEO = "final_output.mp4"

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "model.glb"

print("Loading GLTF safely...")

scene_or_mesh = trimesh.load(MODEL_PATH, force='scene')

meshes = []

if isinstance(scene_or_mesh, trimesh.Scene):
    for name, geom in scene_or_mesh.geometry.items():
        transform = scene_or_mesh.graph.get(name)[0]
        geom = geom.copy()
        geom.apply_transform(transform)

        # 🔥 REMOVE problematic materials
        geom.visual = trimesh.visual.ColorVisuals(
            mesh=geom,
            face_colors=[200, 200, 200, 255]  # gray color
        )

        meshes.append(geom)

    mesh_trimesh = trimesh.util.concatenate(meshes)

else:
    mesh_trimesh = scene_or_mesh

    # 🔥 REMOVE materials
    mesh_trimesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh_trimesh,
        face_colors=[200, 200, 200, 255]
    )

# Convert to pyrender mesh
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

print("✅ Model loaded (materials fixed)")

# =========================
# LOAD FIRST FRAME (for size)
# =========================
frame_files = sorted(os.listdir(FRAME_FOLDER))
first_frame = cv2.imread(os.path.join(FRAME_FOLDER, frame_files[0]))
H, W = first_frame.shape[:2]

# =========================
# VIDEO WRITER
# =========================
writer = imageio.get_writer(OUTPUT_VIDEO, fps=20)

# =========================
# ROTATION MATRIX
# =========================
def get_rotation_matrix(yaw, pitch, roll):
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

# =========================
# MAIN LOOP
# =========================
for idx, file in enumerate(frame_files):

    frame_path = os.path.join(FRAME_FOLDER, file)
    pose_path = os.path.join(POSE_FOLDER, f"pose_{idx:05d}.json")

    frame = cv2.imread(frame_path)
    if frame is None or not os.path.exists(pose_path):
        continue

    with open(pose_path, "r") as f:
        pose = json.load(f)

    # =========================
    # CAMERA SETUP (STATIC)
    # =========================
    FOCAL = pose["camera"]["focal_px"]
    CX = pose["camera"]["cx"]
    CY = pose["camera"]["cy"]

    camera = pyrender.IntrinsicsCamera(
        fx=FOCAL, fy=FOCAL,
        cx=CX, cy=CY
    )

    # =========================
    # CREATE SCENE
    # =========================
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(camera, pose=np.eye(4))

    # =========================
    # LIGHT (IMPORTANT)
    # =========================
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=10.0)
    scene.add(light, pose=np.eye(4))

    # =========================
    # GET POSE
    # =========================
    X = pose["world"]["x"]
    Y = pose["world"]["y"]
    Z = pose["world"]["z"]

    yaw = pose["rotation"]["yaw"]
    pitch = pose["rotation"]["pitch"]
    roll = pose["rotation"]["roll"]  # always 0

    # =========================
    # TRANSFORMATION (FIXED)
    # =========================
    R = get_rotation_matrix(yaw, pitch, roll)

    T = np.eye(4)
    T[:3, :3] = R

    # 🔥 FIX CAMERA AXIS
    T[:3, 3] = [X, -Y, -Z]

    # 🔥 FORCE SCALE (DEBUG)
    scale_factor = 0.2

    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = scale_factor

    final_pose = T @ S
    # =========================
    # SCALE (IMPORTANT)
    # =========================
    scale_factor = 0.2

    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = scale_factor

    final_pose = T @ S

    # =========================
    # ADD MODEL
    # =========================
    scene.add(mesh, pose=final_pose)

    # =========================
    # RENDER
    # =========================
    r = pyrender.OffscreenRenderer(W, H)
    color, depth = r.render(scene)

    print("Center depth:", depth[int(H/2), int(W/2)])
    # =========================
    # COMPOSITING
    # =========================
    mask = depth > 0

    output = frame.copy()
    output[mask] = color[mask]

    writer.append_data(output[:, :, ::-1])  # BGR→RGB

    print(f"Frame {idx} done")

# =========================
# FINALIZE
# =========================
writer.close()
print("✅ Final video saved:", OUTPUT_VIDEO)