import os, sys, time, argparse
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from zhelpers import draw_keypoints, mediapipe_detection  # <-- tus helpers

cv2.setNumThreads(1)  # evita deadlocks en algunos entornos

# ---------- utilidades ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def next_index_path(dirpath, prefix="hola", ext=".npz"):
    ensure_dir(dirpath)
    existing = [f for f in os.listdir(dirpath) if f.startswith(prefix) and f.endswith(ext)]
    nums = []
    for f in existing:
        base = f[len(prefix):-len(ext)]
        try: nums.append(int(base))
        except: pass
    n = (max(nums) + 1) if nums else 1
    return os.path.join(dirpath, f"{prefix}{n:03d}{ext}")

def ema(prev, x, alpha): return (alpha * x + (1 - alpha) * prev) if prev is not None else x

def interpolate_sequence(seq, target_len):
    if len(seq) == target_len: return seq
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    return np.stack([np.interp(x_new, x_old, seq[:, d]) for d in range(seq.shape[1])], axis=1)

def stack_with_derivatives(seq):
    vel = np.diff(seq, axis=0, prepend=seq[0:1])
    acc = np.diff(vel, axis=0, prepend=vel[0:1])
    return np.concatenate([seq, vel, acc], axis=1)

def extract_landmarks(results):
    def lm_to_xyz(lms, has_vis=False):
        if lms is None: return None
        out=[]
        for lm in lms.landmark:
            if has_vis: out.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:       out.extend([lm.x, lm.y, lm.z])
        return np.array(out, dtype=np.float32)

    face = lm_to_xyz(results.face_landmarks, has_vis=False)       # 468*3
    pose = lm_to_xyz(results.pose_landmarks, has_vis=True)        # 33*(3+1)
    lh   = lm_to_xyz(results.left_hand_landmarks, has_vis=False)  # 21*3
    rh   = lm_to_xyz(results.right_hand_landmarks, has_vis=False) # 21*3

    D_face, D_pose, D_hand = 468*3, 33*4, 21*3
    face = face if face is not None else np.zeros(D_face, np.float32)
    pose = pose if pose is not None else np.zeros(D_pose, np.float32)
    lh   = lh   if lh   is not None else np.zeros(D_hand, np.float32)
    rh   = rh   if rh   is not None else np.zeros(D_hand, np.float32)
    return np.concatenate([face, pose, lh, rh], axis=0)

def normalize_by_torso(seq):
    OFF_FACE = 468*3
    D_POSE = 33*4
    if seq.shape[1] < OFF_FACE + D_POSE: return seq
    T, D = seq.shape
    out = seq.copy()
    pose = out[:, OFF_FACE:OFF_FACE+D_POSE].reshape(T, 33, 4)
    l_sh = pose[:, 11, :3]; r_sh = pose[:, 12, :3]
    center = 0.5*(l_sh + r_sh)
    scale = np.linalg.norm(l_sh - r_sh, axis=1)[:, None] + 1e-6
    vis_cols = np.zeros(D, dtype=bool)
    for k in range(33):
        vis_cols[OFF_FACE + k*4 + 3] = True
    for t in range(T):
        i=0
        while i < D:
            if vis_cols[i]:
                i += 1; continue
            for j in range(3):
                if i+j < D and not vis_cols[i+j]:
                    out[t, i+j] = (out[t, i+j] - center[t, j]) / scale[t, 0]
            i += 3
    return out

def save_clip(path, seq, seq_len, fps, label):
    arr = np.stack(seq, axis=0)
    arr = normalize_by_torso(arr)
    arr = interpolate_sequence(arr, seq_len)
    feat = stack_with_derivatives(arr)
    np.savez_compressed(
        path,
        keypoints=arr.astype(np.float32),
        features=feat.astype(np.float32),
        fps=np.array([fps], dtype=np.int32),
        label=np.array([label.upper()]),
        seq_len=np.array([seq_len], dtype=np.int32),
        created=np.array([datetime.now().isoformat()])
    )

# ---------- app ----------
def main():
    ap = argparse.ArgumentParser("LSA one-shot + dibujo (safe mode)")
    ap.add_argument("--label", default="HOLA")
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--prefix", default="hola")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--smooth", type=float, default=0.35)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=["auto","dshow","msmf","v4l2"], default="auto")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--autostop", action="store_true")
    ap.add_argument("--presence_start_frames", type=int, default=5)
    ap.add_argument("--absence_stop_frames", type=int, default=10)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # backend cámara
    backend_flag = 0
    if args.backend == "dshow": backend_flag = cv2.CAP_DSHOW
    elif args.backend == "msmf": backend_flag = cv2.CAP_MSMF
    elif args.backend == "v4l2": backend_flag = cv2.CAP_V4L2

    if backend_flag == 0:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera, backend_flag)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {args.camera} (backend={args.backend}).")
        print("Sugerencias: probar otro --camera (0/1) y en Windows usar --backend dshow.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    label = args.label.upper()
    save_dir = os.path.join(args.out_dir, label)
    save_path = next_index_path(save_dir, prefix=args.prefix.lower(), ext=".npz")

    # Modelo Holistic en modo conservador
    holistic = Holistic(
        static_image_mode=False,
        model_complexity=1,           # bajar complejidad evita cuelgues en PCs justas
        smooth_landmarks=True,
        refine_face_landmarks=True,   # si se cuelga, probá desactivar esto
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Controles: [r]=grabar  [s]=guardar y salir  [ESC/x]=salir sin guardar")
    print(f"Etiqueta: {label}  → próximo archivo: {os.path.basename(save_path)}")
    if args.autostart: print(f"Auto-REC ON ({args.presence_start_frames} frames con manos)")
    if args.autostop:  print(f"Auto-STOP ON ({args.absence_stop_frames} frames sin manos)")

    recording = False
    seq = []
    ema_prev = None
    rec_frames = 0
    presence_count = 0
    absence_count  = 0
    t0 = time.time()
    frames_shown = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No se pudo leer frame.")
                break

            frame = cv2.flip(frame, 1)

            # ---- TU PIPELINE ----
            results = mediapipe_detection(frame, holistic)  # tu helper
            image = frame.copy()
            draw_keypoints(image, results)                  # tu helper

            # estados
            face_ok = results.face_landmarks is not None
            lh_ok   = results.left_hand_landmarks is not None
            rh_ok   = results.right_hand_landmarks is not None
            pose_ok = results.pose_landmarks is not None

            if lh_ok or rh_ok:
                presence_count += 1; absence_count = 0
            else:
                absence_count += 1; presence_count = 0

            # auto‑REC
            if args.autostart and not recording and presence_count >= args.presence_start_frames:
                print("Auto-REC: manos detectadas.")
                recording = True; seq = []; rec_frames = 0

            # features
            vec = extract_landmarks(results)
            ema_prev = ema(ema_prev, vec, args.smooth)
            smoothed = ema_prev.copy()

            if recording:
                seq.append(smoothed)
                rec_frames += 1

            # auto‑STOP
            if args.autostop and recording and absence_count >= args.absence_stop_frames:
                print("Auto-STOP: manos ausentes, guardando clip...")
                if len(seq) > 4:
                    save_clip(save_path, seq, args.seq_len, args.fps, label)
                    print(f"[OK] Guardado: {save_path}")
                else:
                    print("[WARN] Clip demasiado corto, descartado.")
                break

            # overlays
            color = (0,255,0) if recording else (255,255,255)
            cv2.putText(image, f"{label} -> {os.path.basename(save_path)}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image, "REC" if recording else "LISTO", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if recording else (200,200,200), 2)
            cv2.putText(image, f"Frames: {rec_frames if recording else 0}", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            status = f"FACE:{'Y' if face_ok else 'N'}  LH:{'Y' if lh_ok else 'N'}  RH:{'Y' if rh_ok else 'N'}  POSE:{'Y' if pose_ok else 'N'}"
            cv2.putText(image, status, (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

            cv2.imshow('Prototipo_1 (safe)', image)
            frames_shown += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and not recording:
                print("REC manual...")
                recording = True; seq = []; rec_frames = 0
            elif key == ord('s'):
                if recording and len(seq) > 4:
                    print("Guardando clip...")
                    save_clip(save_path, seq, args.seq_len, args.fps, label)
                    print(f"[OK] Guardado: {save_path}")
                else:
                    print("[WARN] Nada que guardar (o clip muy corto).")
                break
            elif key in (27, ord('x')):  # ESC o 'x' como en tu código
                print("Salida sin guardar.")
                break

            # debug FPS
            if args.debug and frames_shown % 30 == 0:
                dt = max(time.time() - t0, 1e-6)
                fps_meas = frames_shown / dt
                print(f"[DEBUG] FPS~{fps_meas:.1f} | recording={recording} | presence={presence_count} absence={absence_count}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            holistic.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
