# %% Entrega 4: CNN congelada como extractor de embeddings + AutoGluon como cabeza de decisión.
# Requiere:  pip install autogluon.tabular
# Ejecutar DESPUÉS de 02_entrenar_profesor.py (usa los mismos splits de resultados/splits.csv).
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config as C, utils as U

df, clases = U.cargar_splits()

# %% 1. Extractor: EfficientNetB0 con include_top=False, trainable=False, pooling='avg' (1280-D)
extractor = keras.applications.EfficientNetB0(include_top=False, weights=C.WEIGHTS, pooling="avg",
                                              input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3))
extractor.trainable = False


def a_tabla(split):
    """Embeddings + etiquetas ALINEADOS: sin shuffle, mismo orden que el DataFrame de splits."""
    d = df[df.split == split].reset_index(drop=True)
    ds = U.dataset(d, split, train=False)            # train=False -> no baraja ni aumenta
    feats = extractor.predict(ds, verbose=0)
    X = pd.DataFrame(feats, columns=[f"f_{i}" for i in range(feats.shape[1])])
    X["clase"] = d.label.values                       # 0..K-1 (multiclase en una sola columna)
    assert len(X) == len(d)
    return X


# %% 2. DataFrames tabulares
df_train, df_val, df_test = a_tabla("train"), a_tabla("val"), a_tabla("test")
print("Tabla train:", df_train.shape)

# %% 3. AutoGluon (TabularPredictor)
from autogluon.tabular import TabularPredictor
multiclase = len(clases) > 2
predictor = TabularPredictor(
    label="clase",
    eval_metric="accuracy" if multiclase else "roc_auc",   # roc_auc solo aplica a binario
    path=f"{C.MODELS_DIR}/autogluon",
).fit(train_data=df_train, tuning_data=df_val, time_limit=300, presets="medium_quality")

# %% 4. Leaderboard (LightGBM, CatBoost, RF, NeuralNet y ensamble) evaluado en TEST
lb = predictor.leaderboard(df_test, silent=True)
print(lb[["model", "score_test", "score_val", "fit_time", "pred_time_test"]])
lb.to_csv(f"{C.RES_DIR}/leaderboard_autogluon.csv", index=False)

# %% 5. Reporte en test
from sklearn.metrics import classification_report, confusion_matrix
y_pred = predictor.predict(df_test.drop(columns="clase"))
print(classification_report(df_test.clase, y_pred, target_names=clases, digits=4))
print(confusion_matrix(df_test.clase, y_pred))
