# %% Entrega 4/5: evaluación en TEST y comparativa Profesor (TL) vs Alumno-KD vs Alumno sin KD
import os, json, time
import numpy as np, pandas as pd, tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from tensorflow import keras
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_recall_fscore_support, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import config as C, utils as U

df, clases = U.cargar_splits()
NC = len(clases)
test = U.dataset(df, "test", False)
y_true = np.concatenate([y.numpy() for _, y in test])
X_test = np.concatenate([x.numpy() for x, _ in test])

modelos = {"Profesor (TL)": C.PROFESOR_PATH, "Alumno + KD": C.ALUMNO_KD_PATH,
           "Alumno sin KD": C.ALUMNO_BASE_PATH}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def tiempo_inferencia(fn, n=100):
    """Latencia media por imagen (batch=1), con calentamiento. Devuelve (ms, FPS)."""
    x = X_test[:1]
    for _ in range(10): fn(x)
    t = time.perf_counter()
    for _ in range(n): fn(x)
    ms = (time.perf_counter() - t) / n * 1000
    return ms, 1000 / ms


# %% 1. Métricas por modelo
filas, probs_dict = [], {}
for nombre, ruta in modelos.items():
    m = keras.models.load_model(ruta)
    f = tf.function(lambda x, m=m: m(x, training=False))
    logits = np.concatenate([m.predict(x, verbose=0) for x, _ in test])
    p = softmax(logits); probs_dict[nombre] = p; y_pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    ms, fps = tiempo_inferencia(lambda x: f(tf.constant(x)).numpy())
    filas.append({"Modelo": nombre, "Accuracy": accuracy_score(y_true, y_pred), "Precision": pr,
                  "Recall": rc, "F1_macro": f1,
                  "ROC_AUC_ovr": roc_auc_score(y_true, p, multi_class="ovr") if NC > 2 else roc_auc_score(y_true, p[:, 1]),
                  "Parametros": m.count_params(), "Tamano_MB": os.path.getsize(ruta) / 1e6,
                  "Latencia_ms": ms, "FPS": fps})
    with open(f"{C.RES_DIR}/reporte_{nombre.split()[0]}_{nombre.split()[-1]}.txt".replace("(", "").replace(")", ""), "w") as fh:
        fh.write(classification_report(y_true, y_pred, target_names=clases, digits=4))
    print("\n==", nombre, "\n", classification_report(y_true, y_pred, target_names=clases, digits=4))

# %% 2. TFLite del alumno (despliegue edge) con cuantización dinámica
alumno = keras.models.load_model(C.ALUMNO_KD_PATH)
conv = tf.lite.TFLiteConverter.from_keras_model(alumno)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open(C.ALUMNO_TFLITE, "wb").write(conv.convert())
itp = tf.lite.Interpreter(model_path=C.ALUMNO_TFLITE); itp.allocate_tensors()
i_in, i_out = itp.get_input_details()[0]["index"], itp.get_output_details()[0]["index"]


def run_tflite(x):
    itp.set_tensor(i_in, x.astype(np.float32)); itp.invoke(); return itp.get_tensor(i_out)


p_tfl = softmax(np.concatenate([run_tflite(X_test[i:i + 1]) for i in range(len(X_test))]))
pr, rc, f1, _ = precision_recall_fscore_support(y_true, p_tfl.argmax(1), average="macro", zero_division=0)
ms, fps = tiempo_inferencia(run_tflite)
filas.append({"Modelo": "Alumno + KD (TFLite int8-dyn)", "Accuracy": accuracy_score(y_true, p_tfl.argmax(1)),
              "Precision": pr, "Recall": rc, "F1_macro": f1,
              "ROC_AUC_ovr": roc_auc_score(y_true, p_tfl, multi_class="ovr") if NC > 2 else roc_auc_score(y_true, p_tfl[:, 1]),
              "Parametros": alumno.count_params(), "Tamano_MB": os.path.getsize(C.ALUMNO_TFLITE) / 1e6,
              "Latencia_ms": ms, "FPS": fps})

res = pd.DataFrame(filas)
res.to_csv(f"{C.RES_DIR}/comparativa.csv", index=False)
print("\n", res.round(4).to_string(index=False))

# trade-off
p, a = res.iloc[0], res.iloc[1]
print(f"\nTRADE-OFF  Δ Accuracy = {(a.Accuracy - p.Accuracy) * 100:+.2f} pp | "
      f"parámetros ÷{p.Parametros / a.Parametros:.0f} | tamaño ÷{p.Tamano_MB / a.Tamano_MB:.0f} | "
      f"velocidad ×{a.FPS / p.FPS:.1f}")

# %% 3. Gráficas: matrices de confusión
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, (n, pp) in zip(ax, probs_dict.items()):
    sns.heatmap(confusion_matrix(y_true, pp.argmax(1)), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=clases, yticklabels=clases, ax=a_)
    a_.set_title(n); a_.set_xlabel("Predicho"); a_.set_ylabel("Real")
plt.tight_layout(); plt.savefig(f"{C.RES_DIR}/matrices_confusion.png", dpi=150); plt.close()

# %% 4. Curvas ROC por clase (OvR)
Yb = label_binarize(y_true, classes=range(NC)) if NC > 2 else None
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, (n, pp) in zip(ax, probs_dict.items()):
    for k in range(NC):
        yk = Yb[:, k] if Yb is not None else (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(yk, pp[:, k]); a_.plot(fpr, tpr, label=f"{clases[k]} (AUC={auc(fpr, tpr):.3f})")
    a_.plot([0, 1], [0, 1], "k--", lw=.8); a_.set_title(f"ROC - {n}"); a_.legend(fontsize=7)
    a_.set_xlabel("FPR"); a_.set_ylabel("TPR")
plt.tight_layout(); plt.savefig(f"{C.RES_DIR}/curvas_roc.png", dpi=150); plt.close()

# %% 5. Comparativa de métricas / eficiencia
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a_, col, t in zip(ax, ["F1_macro", "Parametros", "Tamano_MB", "FPS"],
                      ["F1 macro (test)", "Parámetros (log)", "Tamaño (MB, log)", "FPS (batch=1, CPU)"]):
    a_.bar(range(len(res)), res[col], color=["#37474f", "#2e7d32", "#9e9e9e", "#00897b"])
    a_.set_xticks(range(len(res))); a_.set_xticklabels([m.replace(" (", "\n(") .replace(" + ", "\n+ ") for m in res.Modelo], fontsize=7)
    a_.set_title(t)
    if col in ("Parametros", "Tamano_MB"): a_.set_yscale("log")
plt.tight_layout(); plt.savefig(f"{C.RES_DIR}/comparativa_metricas.png", dpi=150); plt.close()

# %% 6. Curvas de entrenamiento KD vs sin KD
hp = f"{C.RES_DIR}/historial_kd.json"
if os.path.exists(hp):
    h = json.load(open(hp)); plt.figure(figsize=(6, 4))
    for k, lab in (("kd", "Con KD"), ("sin_kd", "Sin KD")):
        plt.plot([e["val_acc"] for e in h[k]], label=f"val_acc {lab}")
    plt.xlabel("Época"); plt.ylabel("Accuracy validación"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{C.RES_DIR}/curvas_entrenamiento.png", dpi=150); plt.close()
print("Resultados en", C.RES_DIR)
