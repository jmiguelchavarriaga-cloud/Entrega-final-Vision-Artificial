"""Utilidades compartidas: splits, pipeline tf.data, modelos Profesor/Alumno."""
import os, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import config as C

EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------- DATOS
def crear_splits():
    """Lee data/dataset/<clase>/*, hace split estratificado y lo guarda en CSV (reproducible)."""
    clases = sorted(d for d in os.listdir(C.DATA_DIR) if os.path.isdir(os.path.join(C.DATA_DIR, d)))
    rows = []
    for i, c in enumerate(clases):
        for f in glob.glob(os.path.join(C.DATA_DIR, c, "*")):
            if f.lower().endswith(EXT):
                rows.append((f, i))
    df = pd.DataFrame(rows, columns=["ruta", "label"])
    tr, tmp = train_test_split(df, test_size=1 - C.SPLIT[0], stratify=df.label, random_state=C.SEED)
    va, te = train_test_split(tmp, test_size=C.SPLIT[2] / (C.SPLIT[1] + C.SPLIT[2]),
                              stratify=tmp.label, random_state=C.SEED)
    tr, va, te = tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")
    pd.concat([tr, va, te]).to_csv(C.SPLITS_CSV, index=False)
    json.dump(clases, open(C.CLASES_JSON, "w"))
    print(f"Clases: {clases} | train={len(tr)} val={len(va)} test={len(te)}")


def cargar_splits():
    df = pd.read_csv(C.SPLITS_CSV)
    clases = json.load(open(C.CLASES_JSON))
    return df, clases


def process_path(ruta, label):
    """Lee y redimensiona. Se deja en [0,255]: EfficientNet normaliza internamente
    y el Alumno trae su propia capa Rescaling -> ambos reciben EXACTAMENTE la misma entrada."""
    img = tf.io.read_file(ruta)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    img.set_shape((C.IMG_SIZE, C.IMG_SIZE, 3))
    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)


_aug = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
                         layers.RandomContrast(0.1)])


def dataset(df, split, train=False, batch=None):
    d = df[df.split == split]
    ds = tf.data.Dataset.from_tensor_slices((d.ruta.values, d.label.values))
    if train:
        ds = ds.shuffle(len(d), seed=C.SEED)
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(lambda x, y: (_aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch or C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- MODELOS
def construir_profesor(num_clases):
    """EfficientNetB0 congelada (include_top=False) + GAP + Dropout + Dense SIN softmax (logits)."""
    base = keras.applications.EfficientNetB0(include_top=False, weights=C.WEIGHTS,
                                             input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3))
    base.trainable = False
    inp = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    logits = layers.Dense(num_clases, name="logits_profesor")(x)
    return keras.Model(inp, logits, name="Profesor_TL")


def construir_alumno(num_clases, nombre="Alumno_Liviano"):
    """CNN compacta: Rescaling + 3 bloques Conv-BN-ReLU-Pool + GAP + Dense (logits)."""
    inp = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_clases, name="logits_alumno")(x)
    return keras.Model(inp, logits, name=nombre)
