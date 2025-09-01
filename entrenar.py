# train_sign_seq.py
import os, glob, argparse, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Utils de carga
# ------------------------------------------------------------
def load_dataset(root_dir):
    """
    Espera estructura:
      root_dir/
        HOLA/*.npz
        COMO_ESTAS/*.npz
    Cada .npz contiene:
      - features: (T, F)  -> ya incluye pos + vel + acc (tu script)
      - label:    [str]   -> nombre de clase
      - seq_len:  [int]   -> longitud re-muestreada (consistente)
    """
    X_list, y_list = [], []
    classes = []
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        classes.append(cls)
        for f in sorted(glob.glob(os.path.join(cls_dir, "*.npz"))):
            try:
                data = np.load(f, allow_pickle=False)
                feat = data["features"]  # (T, F)
                label = cls  # usamos el nombre de carpeta como etiqueta canónica
                X_list.append(feat.astype(np.float32))
                y_list.append(label)
            except Exception as e:
                print(f"[WARN] no pude leer {f}: {e}")
    if not X_list:
        raise RuntimeError("No se encontraron muestras en el dataset.")
    X = np.stack(X_list, axis=0)  # (N, T, F)
    labels = sorted(list(set(y_list)))
    cls_to_idx = {c:i for i,c in enumerate(labels)}
    y = np.array([cls_to_idx[s] for s in y_list], dtype=np.int32)
    return X, y, labels

def train_val_split(X, y, val_ratio=0.2, seed=123):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X)*val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])

def standardize_train(X_train, X_val):
    """
    Estandariza por característica (feature-wise): (x - mean) / std
    Devuelve stats para inferencia.
    """
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-6
    X_train_n = (X_train - mean)/std
    X_val_n   = (X_val   - mean)/std
    return X_train_n, X_val_n, mean, std

# ------------------------------------------------------------
# Modelo: TCN + BiLSTM + Atención ligera
# ------------------------------------------------------------
def temporal_cnn_block(x, filters, kernel_size, dilation):
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    return x

class AttentionPooling(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.w = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, activation=None)

    def call(self, x, mask=None):
        # x: (B, T, H)
        score = self.v(self.w(x))  # (B, T, 1)
        weights = tf.nn.softmax(score, axis=1)  # (B, T, 1)
        if mask is not None:
            # Keras mask: (B, T); aquí no deberíamos necesitarla si T fijo
            pass
        context = tf.reduce_sum(weights * x, axis=1)  # (B, H)
        return context

def build_model(T, F, n_classes):
    inp = layers.Input(shape=(T, F))
    x = temporal_cnn_block(inp, 128, 5, dilation=1)
    x = temporal_cnn_block(x,   128, 5, dilation=2)
    x = temporal_cnn_block(x,   128, 5, dilation=4)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = AttentionPooling(128)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--model_out", default="models/signseq_v1")
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    X, y, labels = load_dataset(args.data_dir)
    (Xtr, ytr), (Xva, yva) = train_val_split(X, y, val_ratio=args.val_ratio)
    Xtr, Xva, mean, std = standardize_train(Xtr, Xva)

    T, F = Xtr.shape[1], Xtr.shape[2]
    n_classes = len(labels)
    print(f"[INFO] N={len(X)} | T={T} | F={F} | clases={labels}")

    model = build_model(T, F, n_classes)
    model.summary()

    os.makedirs(args.model_out, exist_ok=True)

    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=max(2, args.patience//2), verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(args.model_out, "best.keras"),
                                        monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(args.model_out, "training_log.csv"))
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=cbs
    )

    # Guardados para inferencia
    model.save(os.path.join(args.model_out, "final.keras"))
    # guardamos labels y stats de normalización
    with open(os.path.join(args.model_out, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.model_out, "feat_mean.npy"), mean)
    np.save(os.path.join(args.model_out, "feat_std.npy"), std)

    print("[OK] Modelo y artefactos guardados en:", args.model_out)

if __name__ == "__main__":
    main()
