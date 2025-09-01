# extraccion_dinamica.py
# Extracción dinámica de landmarks (cara + pose + manos) con normalización por torso
# y derivadas temporales. Maneja dinámicamente 468 vs 478 puntos de cara (refine).

import os, sys, time, argparse
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from zhelpers import draw_keypoints, mediapipe_detection  # <-- tus helpers

cv2.setNumThreads(0)  # evita deadlocks en algunos entornos

# ---------- utilidades ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def next_index_path(dirpath, prefix="mal", ext=".npz"):
    ensure_dir(dirpath)
    existing = [f for f in os.listdir(dirpath) if f.startswith(prefix) and f.endswith(ext)]
    nums = []
    for f in existing:
        base = f[len(prefix):-len(ext)]
        try: nums.append(int(base))
        except: pass
    n = (max(nums) + 1) if nums else 1
    return os.path.join(dirpath, f"{prefix}{n:03d}{ext}")

def ema(prev, x, alpha): 
    return (alpha * x + (1 - alpha) * prev) if prev is not None else x

def interpolate_sequence(seq, target_len):
    if len(seq) == target_len: 
        return np.asarray(seq, dtype=np.float32)
    seq = np.asarray(seq, dtype=np.float32)
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    # interp columna a columna
    return np.stack([np.interp(x_new, x_old, seq[:, d]) for d in range(seq.shape[1])], axis=1)

def stack_with_derivatives(seq):
    # seq: (T, D)
    vel = np.diff(seq, axis=0, prepend=seq[0:1])
    acc = np.diff(vel, axis=0, prepend=vel[0:1])
    return np.concatenate([seq, vel, acc], axis=1)  # (T, D*3)

# ---------- extracción y normalización ----------
def extract_landmarks(results, refine_face_default=True):
    """
    Devuelve:
      vec: np.ndarray shape (D,) con [face(xyz), pose(xyz+vis), lh(xyz), rh(xyz)]
      face_len: cantidad de puntos de cara usados (468 o 478)
    """
    def lm_to_xyz(lms, has_vis=False):
        if lms is None: 
            return None
        out=[]
        for lm in lms.landmark:
            if has_vis:
                out.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                out.extend([lm.x, lm.y, lm.z])
        return np.array(out, dtype=np.float32)

    face_arr = lm_to_xyz(results.face_landmarks, has_vis=False)
    if face_arr is not None:
        face_len = face_arr.shape[0] // 3  # 468 o 478
    else:
        # Si falla la cara, asumimos según refine
        face_len = 478 if refine_face_default else 468
        face_arr = np.zeros(face_len * 3, np.float32)

    pose_arr = lm_to_xyz(results.pose_landmarks, has_vis=True)
    if pose_arr is None:
        pose_arr = np.zeros(33 * 4, np.float32)

    lh_arr = lm_to_xyz(results.left_hand_landmarks, has_vis=False)
    if lh_arr is None:
        lh_arr = np.zeros(21 * 3, np.float32)

    rh_arr = lm_to_xyz(results.right_hand_landmarks, has_vis=False)
    if rh_arr is None:
        rh_arr = np.zeros(21 * 3, np.float32)

    vec = np.concatenate([face_arr, pose_arr, lh_arr, rh_arr], axis=0)
    return vec, face_len

def normalize_by_torso(seq, face_len):
    """
    Normaliza (x,y,z) restando centro de hombros y escalando por distancia entre hombros.
    No modifica las columnas de 'visibility' del pose.
    Aplica clamp de outliers a [-5, 5].
    """
    OFF_FACE = face_len * 3
    D_POSE = 33 * 4
    if seq.shape[1] < OFF_FACE + D_POSE:
        return seq
    T, D = seq.shape
    out = seq.copy()

    # Pose: (33, 4) = (x,y,z,visibility)
    pose = out[:, OFF_FACE:OFF_FACE + D_POSE].reshape(T, 33, 4)
    l_sh = pose[:, 11, :3]
    r_sh = pose[:, 12, :3]
    center = 0.5 * (l_sh + r_sh)  # (T, 3)
    scale = np.linalg.norm(l_sh - r_sh, axis=1)[:, None] + 1e-6  # (T,1)

    # Marcamos columnas de visibility (cada 4ta del bloque pose)
    vis_cols = np.zeros(D, dtype=bool)
    for k in range(33):
        vis_cols[OFF_FACE + k*4 + 3] = True

    # Recorremos de a (x,y,z) y normalizamos, saltando columnas 'visibility'
    for t in range(T):
        i = 0
        while i < D:
            if vis_cols[i]:
                i += 1
                continue
            # normalizamos triple (x,y,z) si existen
            for j in range(3):
                if i + j < D and not vis_cols[i + j]:
                    out[t, i + j] = (out[t, i + j] - center[t, j]) / scale[t, 0]
            i += 3

    # clamp para evitar outliers extremos
    out = np.clip(out, -5.0, 5.0)
    return out

def save_clip(path, seq, seq_len, fps, label, face_len, subject_id=None, camera_tag=None, lighting=None):
    """
    Guarda:
      - keypoints: (T, D_base_norm)  [ya normalizados por torso]
      - features:  (T, D_base*3)     [pos+vel+acc]
      - meta: fps, label, seq_len, face_len, created, y opcionales subject_id/camera_tag/lighting
    """
    arr = np.stack(seq, axis=0)               # (T0, D_base)
    arr = normalize_by_torso(arr, face_len)   # (T0, D_base)
    arr = interpolate_sequence(arr, seq_len)  # (T, D_base)
    feat = stack_with_derivatives(arr)        # (T, D_base*3)

    meta = {
        "fps": np.array([fps], dtype=np.int32),
        "label": np.array([label.upper()]),
        "seq_len": np.array([seq_len], dtype=np.int32),
        "face_len": np.array([face_len], dtype=np.int32),
        "created": np.array([datetime.now().isoformat()])
    }
    if subject_id is not None:
        meta["subject_id"] = np.array([str(subject_id)])
    if camera_tag is not None:
        meta["camera_tag"] = np.array([str(camera_tag)])
    if lighting is not None:
        meta["lighting"] = np.array([str(lighting)])

    np.savez_compressed(
        path,
        keypoints=arr.astype(np.float32),
        features=feat.astype(np.float32),
        **meta
    )

# ---------- app ----------
def main():
    ap = argparse.ArgumentParser("LSA extractor dinámico (safe, refine-aware)")
    ap.add_argument("--label", default="MAL")
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--prefix", default="mal")
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
    ap.add_argument("--refine_face", action="store_true", help="Activa refine_face_landmarks=True (usa 478 puntos).")
    ap.add_argument("--subject_id", default=None)
    ap.add_argument("--camera_tag", default=None)
    ap.add_argument("--lighting", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # backend cámara
    backend_flag = 0
    if args.backend == "dshow": backend_flag = cv2.CAP_DSHOW
    elif args.backend == "msmf": backend_flag = cv2.CAP_MSMF
    elif args.backend == "v4l2": backend_flag = cv2.CAP_V4L2

    cap = cv2.VideoCapture(args.camera, backend_flag) if backend_flag != 0 else cv2.VideoCapture(args.camera)
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

    # Modelo Holistic
    holistic = Holistic(
        static_image_mode=False,
        model_complexity=1,             # baja complejidad = más estable en equipos modestos
        smooth_landmarks=True,
        refine_face_landmarks=bool(args.refine_face),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Controles: [r]=grabar  [s]=guardar y salir  [ESC/x]=salir sin guardar")
    print(f"Etiqueta: {label}  → próximo archivo: {os.path.basename(save_path)}")
    if args.autostart: print(f"Auto-REC ON ({args.presence_start_frames} frames con manos)")
    if args.autostop:  print(f"Auto-STOP ON ({args.absence_stop_frames} frames sin manos)")
    print(f"refine_face_landmarks = {bool(args.refine_face)} (cara será 478 si está activo)")

    recording = False
    seq = []
    ema_prev = None
    rec_frames = 0
    presence_count = 0
    absence_count  = 0
    face_len_used = None  # se fija cuando llega el primer frame válido
    t0 = time.time()
    frames_shown = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No se pudo leer frame.")
                break

            frame = cv2.flip(frame, 1)

            # ---- tu pipeline con helpers ----
            results = mediapipe_detection(frame, holistic)  # tu helper
            image = frame.copy()
            draw_keypoints(image, results)                  # tu helper

            # estados para overlay
            face_ok = results.face_landmarks is not None
            lh_ok   = results.left_hand_landmarks is not None
            rh_ok   = results.right_hand_landmarks is not None
            pose_ok = results.pose_landmarks is not None

            # presencia/ausencia según manos
            if lh_ok or rh_ok:
                presence_count += 1; absence_count = 0
            else:
                absence_count += 1; presence_count = 0

            # auto-REC
            if args.autostart and not recording and presence_count >= args.presence_start_frames:
                print("Auto-REC: manos detectadas.")
                recording = True; seq = []; rec_frames = 0

            # features por frame
            vec, face_len = extract_landmarks(results, refine_face_default=bool(args.refine_face))
            # fijamos face_len de la secuencia al primer valor confiable
            if face_len_used is None:
                face_len_used = face_len
            elif face_len != face_len_used:
                # no debería ocurrir; avisamos y seguimos usando el primero
                print(f"[WARN] face_len cambió {face_len_used}→{face_len}; mantengo {face_len_used}")

            ema_prev = ema(ema_prev, vec, args.smooth)
            smoothed = ema_prev.copy()

            if recording:
                seq.append(smoothed)
                rec_frames += 1

            # auto-STOP
            if args.autostop and recording and absence_count >= args.absence_stop_frames:
                print("Auto-STOP: manos ausentes, guardando clip...")
                if len(seq) > 4:
                    save_clip(
                        save_path, seq, args.seq_len, args.fps, label, 
                        face_len_used if face_len_used is not None else (478 if args.refine_face else 468),
                        subject_id=args.subject_id, camera_tag=args.camera_tag, lighting=args.lighting
                    )
                    # reporte de dimensiones
                    D_base = (face_len_used if face_len_used is not None else (478 if args.refine_face else 468)) * 3 + 33*4 + 21*3 + 21*3
                    print(f"[OK] Guardado: {save_path} | T={len(seq)} | D_base={D_base} -> features={D_base*3}")
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

            cv2.imshow('Extraccion LSA (safe, refine-aware)', image)
            frames_shown += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and not recording:
                print("REC manual...")
                recording = True; seq = []; rec_frames = 0
            elif key == ord('s'):
                if recording and len(seq) > 4:
                    print("Guardando clip...")
                    save_clip(
                        save_path, seq, args.seq_len, args.fps, label, 
                        face_len_used if face_len_used is not None else (478 if args.refine_face else 468),
                        subject_id=args.subject_id, camera_tag=args.camera_tag, lighting=args.lighting
                    )
                    D_base = (face_len_used if face_len_used is not None else (478 if args.refine_face else 468)) * 3 + 33*4 + 21*3 + 21*3
                    print(f"[OK] Guardado: {save_path} | T={len(seq)} | D_base={D_base} -> features={D_base*3}")
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
