# Visión Artificial · Entrega 4 y 5
**Transfer Learning (EfficientNetB0) → Destilación de Conocimiento por logits → CNN liviana para edge**

Pipeline completo: profesor por Transfer Learning, alumno compacto entrenado con pérdida combinada (CE + KL con temperatura), comparativa de métricas, exportación a TFLite y demo en tiempo real con cámara.

## Estructura
```
src/
  config.py                  # TODO se configura aquí (rutas, tamaño de imagen, alpha, T, épocas)
  utils.py                   # splits, tf.data, modelos Profesor y Alumno
  kd.py                      # pérdida de destilación
  01_generar_dataset_demo.py # (opcional) dataset sintético para probar el pipeline
  02_entrenar_profesor.py    # Entrega 4: Transfer Learning (profesor Keras)
  02b_autogluon_entrega4.py  # Entrega 4: embeddings -> DataFrame f_0..f_n -> AutoGluon + leaderboard
  03_destilacion.py          # Entrega 5: KD  (entrena alumno CON y SIN KD)
  04_evaluacion.py           # métricas, matrices, ROC, comparativa, TFLite
  05_demo_tiempo_real.py     # webcam / video con latencia y FPS
docs/
  BITACORA.md                # repaso teórico de todo lo visto + preguntas tipo del profesor
  GUION_EXPOSICION.md        # reparto de 8-10 min + checklist de la demo
```

## Uso (Spyder o terminal)
1. `pip install -r requirements.txt`
2. Pon tu dataset en `data/dataset/<clase>/*.jpg` (una carpeta por clase). Sin dataset aún: ejecuta `01_generar_dataset_demo.py` para probar.
3. Ajusta `src/config.py` (`IMG_SIZE`, `ALPHA`, `TEMPERATURA`, épocas).
4. Ejecuta en orden `02` → (`02b` opcional, Entrega 4) → `03` → `04` → `05`.

**Spyder:** abre cada script (están divididos en celdas `# %%`; Ctrl+Enter ejecuta la celda). Fija el *working directory* en `src/` (Run → Configuration per file → *Working directory*).
**Terminal:** `cd src && python 02_entrenar_profesor.py` (y así sucesivamente).

Salidas: `models/` (`.keras`, `.tflite`) y `resultados/` (`comparativa.csv`, matrices de confusión, curvas ROC, reportes de clasificación, curvas de entrenamiento).

Prueba rápida sin internet: `KD_WEIGHTS=none KD_EP_PROF=3 KD_EP_ALUM=8 python 02_entrenar_profesor.py` (en Windows PowerShell: `$env:KD_WEIGHTS="none"`).

## Demo en tiempo real
`python 05_demo_tiempo_real.py` (webcam 0) · `--source video.mp4` · `--model keras` · `q` para salir.

## Notas de diseño
- El profesor devuelve **logits** (Dense sin softmax); es la base de la destilación.
- Profesor y alumno reciben la misma entrada en [0,255]; el alumno normaliza internamente (`Rescaling`).
- Se entrena un **alumno sin KD** idéntico como control: la diferencia con el alumno KD mide el aporte real de la destilación.
- Splits estratificados guardados en `resultados/splits.csv` para que todas las métricas usen el mismo test.
- Entrega 4 (AutoGluon): la cabeza tabular consume embeddings `f_0…f_n` de la CNN congelada; ver `docs/BITACORA.md` §8 y el snippet del enunciado. La KD usa la red Keras como profesor porque necesita logits diferenciables por muestra.
