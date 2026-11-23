# %% Comprobación en tiempo real con el alumno destilado (TFLite).
# Uso:  python 05_demo_tiempo_real.py                 -> webcam 0
#       python 05_demo_tiempo_real.py --source video.mp4
#       python 05_demo_tiempo_real.py --model keras   -> usa el .keras en vez de TFLite
# Teclas: q = salir
import argparse, json, time
import cv2, numpy as np, tensorflow as tf
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0"); ap.add_argument("--model", choices=["tflite", "keras"], default="tflite")
ap.add_argument("--max-frames", type=int, default=0, help="0 = infinito (para pruebas sin pantalla)")
ap.add_argument("--headless", action="store_true")
args, _ = ap.parse_known_args()

clases = json.load(open(C.CLASES_JSON))
if args.model == "tflite":
    itp = tf.lite.Interpreter(model_path=C.ALUMNO_TFLITE); itp.allocate_tensors()
    i_in, i_out = itp.get_input_details()[0]["index"], itp.get_output_details()[0]["index"]
    def predecir(x):
        itp.set_tensor(i_in, x); itp.invoke(); return itp.get_tensor(i_out)[0]
else:
    m = tf.keras.models.load_model(C.ALUMNO_KD_PATH)
    predecir = lambda x: m(x, training=False).numpy()[0]

src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit(f"No se pudo abrir la fuente: {src}")

fps_ema, n = 0.0, 0
while True:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    rgb = cv2.cvtColor(cv2.resize(frame, (C.IMG_SIZE, C.IMG_SIZE)), cv2.COLOR_BGR2RGB)  # igual que entrenamiento
    z = predecir(rgb[None].astype(np.float32))
    p = np.exp(z - z.max()); p /= p.sum()
    lat = (time.perf_counter() - t0) * 1000
    fps_ema = 1000 / lat if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1000 / lat)
    k = int(p.argmax())
    cv2.putText(frame, f"{clases[k]}  {p[k]*100:.1f}%", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Latencia {lat:.1f} ms | {fps_ema:.0f} FPS (inferencia)", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
    n += 1
    if args.headless:
        if n % 10 == 0: print(f"frame {n}: {clases[k]} {p[k]*100:.1f}% | {lat:.1f} ms")
    else:
        cv2.imshow("Alumno destilado - tiempo real", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    if args.max_frames and n >= args.max_frames: break
cap.release(); cv2.destroyAllWindows()
