# %% Dataset sintético SOLO para probar el pipeline (círculo / cuadrado / triángulo).
# Si ya tienes tu dataset real, colócalo en data/dataset/<clase>/*.jpg y OMITE este script.
import os, numpy as np, cv2
import config as C

rng = np.random.default_rng(C.SEED)
N = int(os.environ.get("KD_DEMO_N", 200))
for clase in ("circulo", "cuadrado", "triangulo"):
    carpeta = os.path.join(C.DATA_DIR, clase); os.makedirs(carpeta, exist_ok=True)
    for i in range(N):
        img = rng.integers(0, 90, (160, 160, 3), dtype=np.uint8)
        col = tuple(int(v) for v in rng.integers(120, 255, 3))
        cx, cy, r = int(rng.integers(60, 100)), int(rng.integers(60, 100)), int(rng.integers(20, 45))
        if clase == "circulo":
            cv2.circle(img, (cx, cy), r, col, -1)
        elif clase == "cuadrado":
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
        else:
            cv2.fillPoly(img, [np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]])], col)
        cv2.imwrite(os.path.join(carpeta, f"{clase}_{i:03d}.jpg"), img)
print("Dataset demo creado en", C.DATA_DIR)
