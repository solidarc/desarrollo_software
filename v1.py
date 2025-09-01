# live_infer_gated.py
import os, time, argparse
import numpy as np
import cv2
from mediapipe.python.solutions.holistic import Holistic
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== Helpers de tu proyecto (deben existir en zhelpers.py) =====
from zhelpers import draw_keypoints, mediapipe_detection

# ===== Capa personalizada (igual que en entrenamiento) =====
@tf.keras.utils.register_keras_serializable()
class AttentionPooling(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, activation=None)
    def call(self, x, mask=None):
        score = self.v(self.w(x))        # (B, T, 1)
        weights = tf.nn.softmax(score, 1)
        return tf.reduce_sum(weights * x, axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ===== Utilidades numéricas (compatibles con tu dataset) =====
def ema(prev, x, alpha):
    return (alpha * x + (1 - alpha) * prev) if prev is not None else x

def interpolate_sequence(seq, target_len):
    if len(seq) == target_len: return seq
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    return np.vstack([np.interp(x_new, x_old, seq[:, d]) for d in range(seq.shape[1])]).T

def stack_with_derivatives(seq):
    vel = np.diff(seq, axis=0, prepend=seq[0:1])
    acc = np.diff(vel, axis=0, prepend=vel[0:1])
    return np.concatenate([seq, vel, acc], axis=1)

def extract_landmarks(results, want_face_pts=468):
    """
    Devuelve vector (D,) con:
      face (want_face_pts*3) + pose (33*(3+1)) + left hand (21*3) + right hand (21*3).
    Si la cara viene con 478 y pedimos 468, recorta; si viene con 468 y pedimos 478, rellena con ceros.
    """
    def lm_to_xyz(lms, has_vis=False):
        if lms is None: return None
        out=[]
        for lm in lms.landmark:
            if has_vis: out.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:       out.extend([lm.x, lm.y, lm.z])
        return np.array(out, dtype=np.float32)

    # --- FACE ---
    if results.face_landmarks is not None:
        arr = lm_to_xyz(results.face_landmarks, has_vis=False)  # len = n_pts*3
        n_pts = arr.size // 3
        if n_pts == want_face_pts:
            face_vec = arr
        elif n_pts > want_face_pts:
            face_vec = arr[:want_face_pts*3]  # recorta (los 468 primeros preceden al iris)
        else:
            pad = np.zeros((want_face_pts*3 - arr.size,), dtype=np.float32)
            face_vec = np.concatenate([arr, pad], axis=0)
    else:
        face_vec = np.zeros((want_face_pts*3,), dtype=np.float32)

    # --- POSE (33*(xyz+vis)) ---
    pose_vec = lm_to_xyz(results.pose_landmarks, has_vis=True)
    if pose_vec is None:
        pose_vec = np.zeros((33*4,), dtype=np.float32)

    # --- HANDS (21*3 c/u) ---
    lh_vec = lm_to_xyz(results.left_hand_landmarks, has_vis=False)
    rh_vec = lm_to_xyz(results.right_hand_landmarks, has_vis=False)
    if lh_vec is None: lh_vec = np.zeros((21*3,), dtype=np.float32)
    if rh_vec is None: rh_vec = np.zeros((21*3,), dtype=np.float32)

    return np.concatenate([face_vec, pose_vec, lh_vec, rh_vec], axis=0)

def normalize_by_torso(seq, want_face_pts=468):
    """
    Normaliza todo salvo columnas de 'visibility' respecto al centro y escala
    de los hombros (pose 11 y 12). Maneja casos degenerados.
    """
    OFF_FACE = want_face_pts * 3
    D_POSE = 33 * 4
    if seq.shape[1] < OFF_FACE + D_POSE:
        return seq
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
        if scale[t,0] < 1e-5:
            continue
        i=0
        while i < D:
            if vis_cols[i]:
                i += 1; continue
            for j in range(3):
                if i+j < D and not vis_cols[i+j]:
                    out[t, i+j] = (out[t, i+j] - center[t, j]) / scale[t, 0]
            i += 3
    return out

# ===== Cargador de modelo con auto‑ajuste de dimensiones =====
class Predictor:
    def __init__(self, model_dir):
        self.labels = self._load_labels(model_dir)
        self.mean, self.std = self._load_norm(model_dir)
        self.model = self._load_model(model_dir)

        # Dimensión de features esperada por el modelo (F_model = 3 * D_base)
        self.F_model = int(self.mean.shape[-1])  # p.ej., 5076 si entrenaste con iris (478)
        self.D_base_expected = self.F_model // 3  # 1692 si iris; 1662 si sin iris

        # Decomposición de D_base_expected = face(pts*3) + pose(33*4) + hands(2*21*3)
        POSE_DIM = 33*4
        HANDS_DIM = 2*21*3
        self.face_pts_expected = (self.D_base_expected - (POSE_DIM + HANDS_DIM)) // 3
        # -> 478 si modelo con iris; 468 si sin iris

    def _load_model(self, model_dir):
        custom = {"AttentionPooling": AttentionPooling}
        for name in ["best.keras", "final.keras"]:
            path = os.path.join(model_dir, name)
            if os.path.isfile(path):
                try:
                    return keras.models.load_model(path, compile=False, custom_objects=custom)
                except Exception as e:
                    print(f"[WARN] No se pudo cargar {name}: {e}")
        raise FileNotFoundError("No encontré best.keras ni final.keras.")

    def _load_norm(self, model_dir):
        mean = np.load(os.path.join(model_dir, "feat_mean.npy"))
        std  = np.load(os.path.join(model_dir, "feat_std.npy"))
        return mean, std

    def _load_labels(self, model_dir):
        import json
        return json.load(open(os.path.join(model_dir, "label_map.json"), encoding="utf-8"))["labels"]

    def _adjust_face_dim(self, seq_TxD):
        """
        Ajusta SOLO el bloque FACE para que el vector base tenga la dimensión
        esperada por el modelo (468 o 478 pts).
        """
        T, D = seq_TxD.shape
        POSE_DIM = 33*4
        HANDS_DIM = 2*21*3
        face_block_actual = D - (POSE_DIM + HANDS_DIM)
        if face_block_actual % 3 != 0 or face_block_actual <= 0:
            return seq_TxD  # estructura inesperada, no tocamos

        face_pts_actual = face_block_actual // 3
        if face_pts_actual == self.face_pts_expected:
            return seq_TxD  # ya coincide

        OFF_FACE = 0
        OFF_POSE = OFF_FACE + face_pts_actual*3
        OFF_HANDS = OFF_POSE + POSE_DIM

        face = seq_TxD[:, OFF_FACE:OFF_FACE + face_pts_actual*3]
        pose = seq_TxD[:, OFF_POSE:OFF_POSE + POSE_DIM]
        hands = seq_TxD[:, OFF_HANDS:OFF_HANDS + HANDS_DIM]

        if self.face_pts_expected > face_pts_actual:
            # padding con ceros al final del bloque FACE
            pad = np.zeros((T, (self.face_pts_expected - face_pts_actual)*3), dtype=seq_TxD.dtype)
            face_new = np.concatenate([face, pad], axis=1)
        else:
            # truncado a los primeros puntos (los 468 primeros preceden al iris)
            face_new = face[:, :self.face_pts_expected*3]

        seq_new = np.concatenate([face_new, pose, hands], axis=1)
        return seq_new

    def predict_seq(self, seq_TxD, want_face_pts=None):
        """
        seq_TxD: posiciones (T, D) tal como salen del pipeline.
        1) Ajusta FACE a la dimensión esperada por el modelo (468/478).
        2) Normaliza por torso (con la cara ya ajustada).
        3) Deriva features (pos+vel+acc), normaliza y predice.
        """
        # 1) Alinear dimensiones de 'face' al modelo
        seq_aligned = self._adjust_face_dim(seq_TxD)

        # 2) Normalización por torso: necesitamos saber cuántos puntos de cara tiene AHORA
        D_aligned = seq_aligned.shape[1]
        POSE_DIM = 33*4
        HANDS_DIM = 2*21*3
        face_pts_now = (D_aligned - (POSE_DIM + HANDS_DIM)) // 3

        arr = normalize_by_torso(seq_aligned, want_face_pts=face_pts_now)  # (T, D_base)
        feat = stack_with_derivatives(arr)                                  # (T, F)

        # 3) Comprobación final (padding/truncado de features si hiciera falta)
        if feat.shape[-1] != self.F_model:
            Fcur = feat.shape[-1]
            if Fcur < self.F_model:
                pad = np.zeros((feat.shape[0], self.F_model - Fcur), dtype=feat.dtype)
                feat = np.concatenate([feat, pad], axis=1)
            else:
                feat = feat[:, :self.F_model]

        x = (feat[np.newaxis, ...] - self.mean) / (self.std + 1e-6)
        p = self.model.predict(x, verbose=0)[0]
        i = int(p.argmax())
        return self.labels[i], float(p[i]), p

# ===== Máquina de estados con gating por manos =====
IDLE, RECORDING = 0, 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/signseq_v1")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--backend", choices=["auto","dshow","msmf","v4l2"], default="auto")
    ap.add_argument("--smooth", type=float, default=0.35, help="EMA en keypoints")
    ap.add_argument("--presence_start_frames", type=int, default=5, help="frames con >=1 mano para iniciar")
    ap.add_argument("--absence_stop_frames", type=int, default=10, help="frames sin manos para cerrar")
    ap.add_argument("--min_len", type=int, default=16, help="longitud mínima de clip para predecir")
    ap.add_argument("--max_len", type=int, default=120, help="tope de frames por seguridad")
    ap.add_argument("--threshold", type=float, default=0.65, help="confianza mínima")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--refine_face", action="store_true", help="activa iris -> 478 puntos de cara")
    args = ap.parse_args()

    want_face_pts = 478 if args.refine_face else 468

    # cámara
    backend_flag = 0
    if args.backend == "dshow": backend_flag = cv2.CAP_DSHOW
    elif args.backend == "msmf": backend_flag = cv2.CAP_MSMF
    elif args.backend == "v4l2": backend_flag = cv2.CAP_V4L2
    cap = cv2.VideoCapture(args.camera, backend_flag) if backend_flag else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {args.camera} (backend={args.backend})."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # mediapipe
    hol = Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        refine_face_landmarks=bool(args.refine_face),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    predictor = Predictor(args.model_dir)

    state = IDLE
    presence_cnt = 0
    absence_cnt  = 0
    ema_prev = None
    clip = []
    last_pred, last_conf, last_time = "", 0.0, 0.0

    print("Listo! Gating por manos: inicia con presencia, termina con ausencia. [ESC] para salir.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            results = mediapipe_detection(frame, hol)

            # chequeo de manos visibles
            lh_ok = results.left_hand_landmarks is not None
            rh_ok = results.right_hand_landmarks is not None
            any_hand = lh_ok or rh_ok

            # landmarks + suavizado
            vec = extract_landmarks(results, want_face_pts=want_face_pts)
            ema_prev = ema(ema_prev, vec, args.smooth)

            # lógica de presencia/ausencia
            if any_hand:
                presence_cnt += 1; absence_cnt = 0
            else:
                absence_cnt += 1; presence_cnt = 0

            # máquina de estados
            if state == IDLE:
                if presence_cnt >= args.presence_start_frames:
                    state = RECORDING
                    clip = []
            elif state == RECORDING:
                clip.append(ema_prev.copy())
                if len(clip) >= args.max_len:
                    absence_cnt = max(absence_cnt, args.absence_stop_frames)  # fuerza cierre por tope
                if absence_cnt >= args.absence_stop_frames:
                    # cerrar clip y predecir si es suficientemente largo
                    if len(clip) >= args.min_len:
                        seq = np.stack(clip, axis=0)
                        seq = interpolate_sequence(seq, args.seq_len)
                        label, conf, _ = predictor.predict_seq(seq, want_face_pts=want_face_pts)
                        if conf >= args.threshold:
                            last_pred, last_conf, last_time = label, conf, time.time()
                        else:
                            last_pred, last_conf, last_time = "(baja confianza)", conf, time.time()
                    # reset
                    clip = []
                    state = IDLE

            # dibujo/overlay
            image = frame.copy()
            draw_keypoints(image, results)
            cv2.putText(image, f"Estado: {'Tomando' if state==RECORDING else 'IDLE'}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255) if state==RECORDING else (200,200,200), 2)
            cv2.putText(image, f"Manos: {'Y' if any_hand else 'N'}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if any_hand else (0,0,255), 2)
            if (time.time() - last_time) < 2.0 and last_pred:
                cv2.putText(image, f"{last_pred}  {last_conf:.0%}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if last_pred!='(baja confianza)' else (0,200,255), 2)
            if state == RECORDING:
                cv2.putText(image, f"Clip len: {len(clip)}", (10, 104),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Live Sign Inference (Hand-gated)", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break  # ESC
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try: hol.close()
        except: pass

if __name__ == "__main__":
    # (opcional) manejo de memoria GPU
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass
    main()
