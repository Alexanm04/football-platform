"""
Diagnóstico rápido — procesa solo los primeros N frames y mide tiempos
exactos de decode vs inferencia, para localizar el cuello de botella
en segundos en vez de esperar horas.

Uso:
    python diagnose.py /ruta/a/tu/video.mp4
"""
import sys
import time

import cv2
import torch
from ultralytics import YOLO

N_FRAMES = 30  

def main(video_path):
    print(f"PyTorch: {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")

    t_load_start = time.time()
    model = YOLO("yolov8m.pt").to(device)
    if device == "cuda":
        model.model.half()
    print(f"Modelo cargado en {time.time()-t_load_start:.2f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir el vídeo. Revisa la ruta.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vídeo: {w}x{h} @ {fps}fps, {total} frames totales")

    decode_times = []
    infer_times = []

    ret, frame = cap.read()
    if not ret:
        print("ERROR: no se pudo leer ni el primer frame.")
        return
    print("Ejecutando warm-up (primera inferencia, no cuenta para la media)...")
    t0 = time.time()
    _ = model.track(frame, classes=[0, 32], persist=True, tracker="bytetrack.yaml",
                     conf=0.25, iou=0.5, imgsz=640, verbose=False, half=(device == "cuda"))
    print(f"Warm-up: {time.time()-t0:.2f}s (normal que sea más lento que el resto)")

    print(f"\nMidiendo {N_FRAMES} frames reales...")
    for i in range(N_FRAMES):
        t_r0 = time.time()
        ret, frame = cap.read()
        t_r1 = time.time()
        if not ret:
            print(f"Vídeo terminó en el frame {i}")
            break
        decode_times.append(t_r1 - t_r0)

        t_i0 = time.time()
        _ = model.track(frame, classes=[0, 32], persist=True, tracker="bytetrack.yaml",
                         conf=0.25, iou=0.5, imgsz=640, verbose=False, half=(device == "cuda"))
        t_i1 = time.time()
        infer_times.append(t_i1 - t_i0)

        print(f"  frame {i+1}/{N_FRAMES}: decode={1000*(t_r1-t_r0):.0f}ms  infer={1000*(t_i1-t_i0):.0f}ms")

    cap.release()

    if decode_times:
        avg_decode = sum(decode_times) / len(decode_times)
        avg_infer = sum(infer_times) / len(infer_times)
        print(f"\n--- RESUMEN ---")
        print(f"Decode promedio:     {1000*avg_decode:.0f}ms/frame")
        print(f"Inferencia promedio: {1000*avg_infer:.0f}ms/frame")
        print(f"Total por frame:     {1000*(avg_decode+avg_infer):.0f}ms/frame")
        print(f"Estimación para {total} frames: {(avg_decode+avg_infer)*total/60:.1f} minutos "
              f"(solo decode+inferencia, sin contar tracker/postproc/render)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnose.py /ruta/a/tu/video.mp4")
        sys.exit(1)
    main(sys.argv[1])
