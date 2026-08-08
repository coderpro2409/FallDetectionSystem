import os
import cv2
import mediapipe as mp
import numpy as np
import random
import argparse

DATASET_DIR = "dataset"
OUTPUT_DIR = "extracted_features"
MAX_TARGET_FALL_SAMPLES = 100
MAX_TARGET_NORMAL_SAMPLES = 100

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calc_angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180.0 else ang

def extract_invariant_features(landmarks):
    nose = landmarks[0]; l_sh = landmarks[11]; r_sh = landmarks[12]
    l_el = landmarks[13]; r_el = landmarks[14]; l_wr = landmarks[15]; r_wr = landmarks[16]
    l_hip = landmarks[23]; r_hip = landmarks[24]
    l_knee = landmarks[25]; r_knee = landmarks[26]
    l_ankle = landmarks[27]; r_ankle = landmarks[28]
    l_foot = landmarks[31]; r_foot = landmarks[32]

    head_tilt = calc_angle(l_sh, nose, r_sh)
    l_sh_angle = calc_angle(r_sh, l_sh, nose)
    r_sh_angle = calc_angle(l_sh, r_sh, nose)
    l_elbow = calc_angle(l_sh, l_el, l_wr)
    r_elbow = calc_angle(r_sh, r_el, r_wr)

    if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
        l_hip_angle = calc_angle(l_sh, l_hip, l_knee)
        r_hip_angle = calc_angle(r_sh, r_hip, r_knee)
        l_knee_angle = calc_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calc_angle(r_hip, r_knee, r_ankle)
        l_ankle_angle = calc_angle(l_knee, l_ankle, l_foot)
        r_ankle_angle = calc_angle(r_knee, r_ankle, r_foot)
    else:
        l_hip_angle = r_hip_angle = l_knee_angle = r_knee_angle = l_ankle_angle = r_ankle_angle = 0

    sh_vec = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y])
    sh_line_angle = np.arctan2(sh_vec[1], sh_vec[0]) * 180.0 / np.pi
    
    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2])
        mid_hip = np.array([(l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2])
        torso_angle = np.arctan2((mid_hip - mid_sh)[1], (mid_hip - mid_sh)[0]) * 180.0 / np.pi
        hip_line_angle = np.arctan2((r_hip.x - l_hip.x), (r_hip.y - l_hip.y)) * 180.0 / np.pi
    else:
        torso_angle = 0; hip_line_angle = 0

    shoulder_width = np.linalg.norm(sh_vec)
    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2])
        mid_hip = np.array([(l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2])
        torso_len = np.linalg.norm(mid_sh - mid_hip)
        l_leg_len = np.linalg.norm(np.array([l_hip.x, l_hip.y]) - np.array([l_ankle.x, l_ankle.y]))
        r_leg_len = np.linalg.norm(np.array([r_hip.x, r_hip.y]) - np.array([r_ankle.x, r_ankle.y]))
        avg_leg_len = (l_leg_len + r_leg_len) / 2
        leg_torso_ratio = avg_leg_len / torso_len if torso_len > 0.001 else 0
    else:
        leg_torso_ratio = 0

    return np.array([head_tilt, l_sh_angle, r_sh_angle, l_elbow, r_elbow,
                     l_hip_angle, r_hip_angle, l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle,
                     torso_angle, sh_line_angle, hip_line_angle, shoulder_width, leg_torso_ratio])

def rotate_point(x, y, cx, cy, angle_deg):
    rad = np.radians(angle_deg)
    x -= cx; y -= cy
    return x * np.cos(rad) - y * np.sin(rad) + cx, x * np.sin(rad) + y * np.cos(rad) + cy

def augment_landmarks(landmarks):
    augmented_list = [landmarks]
    cx = (landmarks[11].x + landmarks[12].x) / 2
    cy = (landmarks[11].y + landmarks[12].y) / 2
    
    # KEEP ONLY PHYSICALLY ACCURATE ROTATIONS. Remove foreshortening.
    for angle in [-15, 15]:
        new_landmarks = []
        for lm in landmarks:
            nx, ny = rotate_point(lm.x, lm.y, cx, cy, angle)
            new_lm = type('Landmark', (object,), {'x': nx, 'y': ny, 'z': lm.z, 'visibility': lm.visibility})()
            new_landmarks.append(new_lm)
        augmented_list.append(new_landmarks)
    return augmented_list

def get_label_from_filename(filename):
    return 'Falling' if 'fall' in filename.lower() else 'Normal'

def find_annotation_file(video_path):
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    potential_ann = os.path.join(base_dir, base_name + '.txt')
    if os.path.exists(potential_ann): return potential_ann
    if os.path.basename(base_dir) == 'Videos':
        ann_dir = os.path.join(os.path.dirname(base_dir), 'Annotation_files')
        potential_ann = os.path.join(ann_dir, base_name + '.txt')
        if os.path.exists(potential_ann): return potential_ann
    return None

def parse_le2i_end_frame(ann_path):
    try:
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2: return int(lines[1].strip())
    except: pass
    return 0

def extract_features_from_video(video_path, frame_num, label):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret: return False
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        for aug_lm in augment_landmarks(res.pose_landmarks.landmark):
            feats = extract_invariant_features(aug_lm)
            base_filename = f"{label}_{os.path.basename(video_path)}_frame_{frame_num}_rot"
            npy_path = os.path.join(OUTPUT_DIR, base_filename + ".npy")
            np.save(npy_path, feats)
        return True
    return False

def extract_features():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fall_pool, normal_pool = [], []
    
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                vid_path = os.path.join(root, file)
                ann_path = find_annotation_file(vid_path)
                if ann_path:
                    end_frame = parse_le2i_end_frame(ann_path)
                    if end_frame > 0: fall_pool.append((vid_path, end_frame))
                    else: normal_pool.append((vid_path, get_label_from_filename(file)))
    
    random.shuffle(fall_pool); random.shuffle(normal_pool)
    fall_extracted, normal_extracted = 0, 0

    for path, end_frame in fall_pool:
        if fall_extracted >= MAX_TARGET_FALL_SAMPLES: break
        for i in range(5):
            if fall_extracted >= MAX_TARGET_FALL_SAMPLES: break
            frame_num = end_frame - i
            if frame_num <= 0: break
            if extract_features_from_video(path, frame_num, "Falling"): fall_extracted += 1
    print(f"Extracted {fall_extracted} base Fall frames. (Rot Augmented)")

    for path, label in normal_pool:
        if normal_extracted >= MAX_TARGET_NORMAL_SAMPLES: break
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames == 0: continue
        take_this = min(8, total_frames, MAX_TARGET_NORMAL_SAMPLES - normal_extracted)
        if take_this > 0:
            for f_num in random.sample(range(0, total_frames), take_this):
                if extract_features_from_video(path, f_num, label):
                    normal_extracted += 1
    print(f"Extracted {normal_extracted} base Normal frames. (Rot Augmented)")

if __name__ == "__main__":
    extract_features()