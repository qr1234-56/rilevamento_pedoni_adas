"""
Misura del tempo di elaborazione in tempo reale con la webcam.

La modalita' webcam dell'interfaccia non salva statistiche: questo script
riproduce la stessa catena (acquisizione a 640x480, PedestrianDetector con
soglia 0,25, codifica JPEG del fotogramma annotato) e registra per ogni
fotogramma il tempo impiegato. Il sistema non viene modificato.

Esecuzione dalla cartella principale del progetto, con la webcam libera
(interfaccia web chiusa):

    python tools/misura_webcam.py            # 60 secondi
    python tools/misura_webcam.py --durata 90
"""

import argparse
import csv
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.detector import PedestrianDetector

PROGETTO = Path(__file__).resolve().parent.parent


def processore():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--durata", type=float, default=60.0, help="secondi di misura")
    pa.add_argument("--camera", type=int, default=0)
    args = pa.parse_args()

    detector = PedestrianDetector(confidence_threshold=0.25)
    detector.audio_alert.play = lambda *a, **k: None
    detector.reset()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Webcam non disponibile")
        return

    # Riscaldamento: i primi fotogrammi includono il caricamento del modello.
    for _ in range(20):
        ok, frame = cap.read()
        if ok:
            detector.process(frame)
    detector.reset()

    print(f"Misura in corso per {args.durata:.0f} s: muoviti davanti alla webcam...")
    righe = []
    inizio = time.perf_counter()
    precedente = inizio
    while time.perf_counter() - inizio < args.durata:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = detector.process(frame)
        cv2.imencode(".jpg", r["frame"])
        t1 = time.perf_counter()
        righe.append({
            "tempo_s": round(t1 - inizio, 3),
            "elaborazione_ms": round((t1 - t0) * 1000, 2),
            "intervallo_ms": round((t1 - precedente) * 1000, 2),
            "pedoni": r["pedestrians"],
            "livello": r["alert_level"],
        })
        precedente = t1
    cap.release()

    uscita = PROGETTO / "results" / "webcam_tempi.csv"
    with open(uscita, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    el = [r["elaborazione_ms"] for r in righe]
    iv = [r["intervallo_ms"] for r in righe[1:]]
    con_pedoni = [r["elaborazione_ms"] for r in righe if r["pedoni"] > 0]
    print()
    print(f"processore: {processore()}")
    print(f"fotogrammi: {len(righe)} in {righe[-1]['tempo_s']:.1f} s "
          f"-> {len(righe) / righe[-1]['tempo_s']:.1f} fps effettivi")
    print(f"elaborazione per fotogramma: media {statistics.mean(el):.1f} ms, "
          f"mediana {statistics.median(el):.1f} ms, min {min(el):.1f}, max {max(el):.1f}")
    print(f"  -> {1000 / statistics.mean(el):.1f} fps possibili in base alla sola elaborazione")
    if con_pedoni:
        print(f"fotogrammi con pedoni: {len(con_pedoni)}, media {statistics.mean(con_pedoni):.1f} ms")
    print(f"intervallo fra fotogrammi: media {statistics.mean(iv):.1f} ms, max {max(iv):.1f} ms")
    print(f"dettaglio in {uscita}")


if __name__ == "__main__":
    main()
