# %% Entrega 4: Transfer Learning (profesor). Backbone congelado + cabeza densa.
import numpy as np, tensorflow as tf
from tensorflow import keras
import config as C, utils as U

tf.random.set_seed(C.SEED); np.random.seed(C.SEED)
# %% 1. Splits reproducibles
U.crear_splits()
df, clases = U.cargar_splits()
train, val = U.dataset(df, "train", True), U.dataset(df, "val")

# %% 2. Entrenar la cabeza (backbone congelado)
prof = U.construir_profesor(len(clases))
prof.compile(optimizer=keras.optimizers.Adam(C.LR_PROFESOR),
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
h = prof.fit(train, validation_data=val, epochs=C.EPOCHS_PROFESOR)

# %% 3. (Opcional) Fine-tuning de las últimas capas del backbone
if C.EPOCHS_FINE_TUNE > 0:
    base = prof.layers[1]; base.trainable = True
    for l in base.layers[:-20]: l.trainable = False
    prof.compile(optimizer=keras.optimizers.Adam(1e-5),
                 loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    prof.fit(train, validation_data=val, epochs=C.EPOCHS_FINE_TUNE)

# %% 4. Guardar
prof.save(C.PROFESOR_PATH)
print("Profesor guardado en", C.PROFESOR_PATH)
