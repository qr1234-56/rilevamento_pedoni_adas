"""
Salva in JSON le predizioni di yolov8n su alcune immagini KITTI di
validazione, prima e dopo la Non-Maximum Suppression, per le figure
del capitolo 2. Non modifica nulla del sistema.

Uso: .venv/bin/python tools/predizioni_figure.py
"""
import json
from pathlib import Path

from ultralytics import YOLO

PROGETTO = Path(__file__).resolve().parent.parent
IMMAGINI = ["003753", "006980", "006395", "004532", "005510"]
CARTELLA = PROGETTO / "dataset" / "kitti" / "images" / "val"
USCITA = PROGETTO / "figure_capitolo_2" / "predizioni.json"


def riquadri(risultato):
    b = risultato.boxes
    return [
        {"xyxy": [round(v, 1) for v in xyxy], "conf": round(c, 3)}
        for xyxy, c in zip(b.xyxy.tolist(), b.conf.tolist())
    ]


modello = YOLO(str(PROGETTO / "yolov8n.pt"))
dati = {}
for nome in IMMAGINI:
    img = str(CARTELLA / f"{nome}.png")
    # iou=1.0: la NMS non elimina nulla, restano tutte le predizioni grezze
    grezze = modello.predict(img, conf=0.25, iou=1.0, classes=[0],
                             max_det=1000, verbose=False)[0]
    # impostazione del sistema: NMS con soglia IoU 0,7
    finali = modello.predict(img, conf=0.25, iou=0.7, classes=[0],
                             verbose=False)[0]
    dati[nome] = {"prima_nms": riquadri(grezze), "dopo_nms": riquadri(finali)}
    print(nome, len(dati[nome]["prima_nms"]), "->", len(dati[nome]["dopo_nms"]))

USCITA.parent.mkdir(exist_ok=True)
USCITA.write_text(json.dumps(dati, indent=1))
print("salvato in", USCITA)
