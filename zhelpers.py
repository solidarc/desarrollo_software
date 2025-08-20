import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

# Detecta los keypoints con MediaPipe
def mediapipe_detection(imagen, model):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    results = model.process(imagen)
    return results


# Crea carpeta si no existe
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Verifica si hay alguna mano detectada
def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks


# Dibuja los puntos clave en la imagenn
def draw_keypoints(imagen, results):
    if results.face_landmarks:
        draw_landmarks(
            imagen,
            results.face_landmarks,
            FACEMESH_CONTOURS,
            DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        draw_landmarks(
            imagen,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        draw_landmarks(
            imagen,
            results.left_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        draw_landmarks(
            imagen,
            results.right_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

# Guarda cada frame como imagen en la carpeta destino
def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z ]for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z ] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z ] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
