"""Configuración central del proyecto. Edita SOLO este archivo para adaptar al problema."""
import os

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(RAIZ, "data", "dataset")      # data/dataset/<clase>/*.jpg
MODELS_DIR = os.path.join(RAIZ, "models")
RES_DIR = os.path.join(RAIZ, "resultados")
for _d in (MODELS_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

IMG_SIZE = 128                 # lado de la imagen (128 rápido; 224/256 si tu dataset lo pide)
BATCH_SIZE = 32
SEED = 42
SPLIT = (0.70, 0.15, 0.15)     # train / val / test (estratificado)

# --- Profesor (Transfer Learning) ---
# "imagenet" requiere internet la primera vez. Para pruebas offline: export KD_WEIGHTS=none
WEIGHTS = None if os.environ.get("KD_WEIGHTS", "imagenet").lower() == "none" else "imagenet"
EPOCHS_PROFESOR = int(os.environ.get("KD_EP_PROF", 8))
EPOCHS_FINE_TUNE = 0           # >0 descongela las últimas capas del backbone
LR_PROFESOR = 1e-3

# --- Destilación ---
EPOCHS_ALUMNO = int(os.environ.get("KD_EP_ALUM", 25))
ALPHA = 0.3                    # peso de la etiqueta real (1-ALPHA = peso del profesor)
TEMPERATURA = 4.0
LR_ALUMNO = 1e-3

# --- Rutas de salida ---
SPLITS_CSV = os.path.join(RES_DIR, "splits.csv")
CLASES_JSON = os.path.join(RES_DIR, "clases.json")
PROFESOR_PATH = os.path.join(MODELS_DIR, "profesor_tl.keras")
ALUMNO_KD_PATH = os.path.join(MODELS_DIR, "alumno_kd.keras")
ALUMNO_BASE_PATH = os.path.join(MODELS_DIR, "alumno_sin_kd.keras")
ALUMNO_TFLITE = os.path.join(MODELS_DIR, "alumno_kd.tflite")
