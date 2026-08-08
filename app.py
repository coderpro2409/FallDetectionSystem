import streamlit as st
import cv2, os, numpy as np, mediapipe as mp, joblib
from PIL import Image
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

st.set_page_config(page_title="AI Elderly Fall Detection", layout="wide")

@st.cache_resource
def load_rf_model():
    return joblib.load('fall_detection_rf.pkl')

@st.cache_resource
def load_landmarker():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path): st.error("Missing 'pose_landmarker.task'"); st.stop()
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            min_pose_detection_confidence=0.5
        )
    )

rf_model = load_rf_model()
landmarker = load_landmarker()
CLASS_LABELS = {0: "⚠️ FALL DETECTED", 1: "✅ Normal Activity"}
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def calc_angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180.0 else ang

def extract_invariant_features(landmarks):
    nose = landmarks[0]; l_sh = landmarks[11]; r_sh = landmarks[12]
    l_el = landmarks[13]; r_el = landmarks[14]
    l_wr = landmarks[15]; r_wr = landmarks[16]
    l_hip = landmarks[23]; r_hip = landmarks[24]
    l_knee = landmarks[25]; r_knee = landmarks[26]
    l_ankle = landmarks[27]; r_ankle = landmarks[28]
    l_foot = landmarks[31]; r_foot = landmarks[32]

    head_tilt = calc_angle(l_sh, nose, r_sh)
    l_sh_angle = calc_angle(r_sh, l_sh, nose); r_sh_angle = calc_angle(l_sh, r_sh, nose)
    l_elbow = calc_angle(l_sh, l_el, l_wr); r_elbow = calc_angle(r_sh, r_el, r_wr)

    if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
        l_hip_angle, r_hip_angle, l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle = \
            calc_angle(l_sh, l_hip, l_knee), calc_angle(r_sh, r_hip, r_knee), \
            calc_angle(l_hip, l_knee, l_ankle), calc_angle(r_hip, r_knee, r_ankle), \
            calc_angle(l_knee, l_ankle, l_foot), calc_angle(r_knee, r_ankle, r_foot)
    else: l_hip_angle = r_hip_angle = l_knee_angle = r_knee_angle = l_ankle_angle = r_ankle_angle = 0

    sh_vec = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y])
    sh_line_angle = np.arctan2(sh_vec[1], sh_vec[0]) * 180.0 / np.pi
    
    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2])
        mid_hip = np.array([(l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2])
        torso_angle = np.arctan2((mid_hip - mid_sh)[1], (mid_hip - mid_sh)[0]) * 180.0 / np.pi
        hip_line_angle = np.arctan2((r_hip.x - l_hip.x), (r_hip.y - l_hip.y)) * 180.0 / np.pi
    else: torso_angle, hip_line_angle = 0, 0

    shoulder_width = np.linalg.norm(sh_vec)
    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2])
        mid_hip = np.array([(l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2])
        torso_len = np.linalg.norm(mid_sh - mid_hip)
        avg_leg_len = (np.linalg.norm(np.array([l_hip.x, l_hip.y]) - np.array([l_ankle.x, l_ankle.y])) + 
                       np.linalg.norm(np.array([r_hip.x, r_hip.y]) - np.array([r_ankle.x, r_ankle.y]))) / 2
        leg_torso_ratio = avg_leg_len / torso_len if torso_len > 0.001 else 0
    else: leg_torso_ratio = 0

    return np.array([head_tilt, l_sh_angle, r_sh_angle, l_elbow, r_elbow, l_hip_angle, r_hip_angle,
                     l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle, torso_angle, 
                     sh_line_angle, hip_line_angle, shoulder_width, leg_torso_ratio]).reshape(1, -1)

def draw_landmarks_on_image(image, result):
    if not result.pose_landmarks: return image
    h, w, _ = image.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks[0]]
    for conn in POSE_CONNECTIONS:
        if conn[0] < len(pts) and conn[1] < len(pts):
            cv2.line(image, pts[conn[0]], pts[conn[1]], (0, 255, 0), 2)
    for pt in pts: cv2.circle(image, pt, 3, (0, 0, 255), -1)
    return image

st.title("🏥 AI Elderly Fall Detection (Random Forest)")
if 'fall' not in st.session_state: st.session_state.fall = 0
if 'normal' not in st.session_state: st.session_state.normal = 0

input_mode = st.radio("Input source", ["📁 Upload Image", "📷 Live Camera"], horizontal=True)
uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"]) if input_mode == "📁 Upload Image" else st.camera_input("Take a photo")

col1, col2 = st.columns([2, 1])

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    annotated_img = draw_landmarks_on_image(img.copy(), result)
    
    if result.pose_landmarks:
        feats = extract_invariant_features(result.pose_landmarks[0])
        # Use predict_proba for probabilities
        pred_probs = rf_model.predict_proba(feats)
        pred_idx = np.argmax(pred_probs[0])
        conf = np.max(pred_probs[0])
        
        if pred_idx == 0:
            st.session_state.fall += 1
            col2.error("🚨 EMERGENCY ALERT: FALL DETECTED!")
            cv2.rectangle(annotated_img, (0,0), (annotated_img.shape[1], annotated_img.shape[0]), (0,0,255), 8)
            cv2.putText(annotated_img, "FALL", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        else:
            st.session_state.normal += 1
            col2.success(f"✅ {CLASS_LABELS[pred_idx]} ({conf:.2f})")
            cv2.putText(annotated_img, CLASS_LABELS[pred_idx], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        col1.image(annotated_img, use_column_width=True)
    else: col1.warning("No pose detected.")

with col2:
    total = st.session_state.fall + st.session_state.normal
    st.metric("Total", total); st.metric("Falls", st.session_state.fall); st.metric("Normals", st.session_state.normal)
    if os.path.exists("evaluation_plots"):
        st.image("evaluation_plots/accuracy_loss_graphs.png", use_column_width=True) # Note: RF doesn't have loss graphs, but keep it for UI
        st.image("evaluation_plots/confusion_matrix.png", use_column_width=True)