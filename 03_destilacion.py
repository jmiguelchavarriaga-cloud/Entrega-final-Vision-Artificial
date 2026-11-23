# %% Entrega 5: Destilación de conocimiento por logits (Profesor congelado -> Alumno liviano)
# Entrena DOS alumnos idénticos: con KD (alpha<1) y sin KD (alpha=1, solo etiqueta real)
# para demostrar cuánto aporta el profesor.
import json, numpy as np, tensorflow as tf
from tensorflow import keras
import config as C, utils as U
from kd import perdida_kd

tf.random.set_seed(C.SEED); np.random.seed(C.SEED)
df, clases = U.cargar_splits()
NC = len(clases)
train, val = U.dataset(df, "train", True), U.dataset(df, "val")

profesor = keras.models.load_model(C.PROFESOR_PATH)
profesor.trainable = False                      # el profesor NUNCA se entrena aquí


def entrenar_alumno(alpha, T, nombre, ruta):
    alumno = U.construir_alumno(NC, nombre)
    opt = keras.optimizers.Adam(C.LR_ALUMNO)

    @tf.function
    def paso(x, y):
        lp = profesor(x, training=False)
        with tf.GradientTape() as tape:
            ls = alumno(x, training=True)
            total, l_real, l_kd = perdida_kd(y, ls, lp, alpha, T)
        g = tape.gradient(total, alumno.trainable_variables)
        opt.apply_gradients(zip(g, alumno.trainable_variables))
        acc = tf.reduce_mean(tf.cast(tf.argmax(ls, -1, output_type=tf.int32) == y, tf.float32))
        return total, l_real, l_kd, acc

    @tf.function
    def paso_val(x, y):
        ls = alumno(x, training=False)
        return tf.reduce_mean(tf.cast(tf.argmax(ls, -1, output_type=tf.int32) == y, tf.float32))

    hist, mejor = [], -1
    for ep in range(1, C.EPOCHS_ALUMNO + 1):
        m = np.array([[v.numpy() for v in paso(x, y)] for x, y in train]).mean(0)
        va = float(np.mean([paso_val(x, y).numpy() for x, y in val]))
        hist.append(dict(epoca=ep, loss=m[0], loss_real=m[1], loss_kd=m[2], acc=m[3], val_acc=va))
        if va >= mejor:
            mejor = va; alumno.save(ruta)
        print(f"[{nombre}] ep {ep:02d} loss={m[0]:.4f} (real {m[1]:.3f} | kd {m[2]:.3f}) "
              f"acc={m[3]*100:.1f}% val_acc={va*100:.1f}%")
    return hist


# %% Alumno CON KD
h_kd = entrenar_alumno(C.ALPHA, C.TEMPERATURA, "Alumno_KD", C.ALUMNO_KD_PATH)
# %% Alumno SIN KD (baseline: alpha=1 -> solo cross-entropy con etiqueta real)
h_base = entrenar_alumno(1.0, 1.0, "Alumno_SinKD", C.ALUMNO_BASE_PATH)

json.dump({"kd": h_kd, "sin_kd": h_base}, open(f"{C.RES_DIR}/historial_kd.json", "w"), default=float)
print("Destilación completada.")
