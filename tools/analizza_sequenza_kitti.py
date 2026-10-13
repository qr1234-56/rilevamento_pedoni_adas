"""
Analisi del sistema completo su una sequenza grezza di KITTI (raw data).

Le sequenze grezze sono fotogrammi consecutivi a circa 10 fps con i punti
del lidar e i dati GPS/inerziali del veicolo. Lo script fa passare ogni
fotogramma della telecamera a colori sinistra (image_02) attraverso il
PedestrianDetector del sistema, con il tempo reale di acquisizione, e per
ogni pedone tracciato registra:

  - i valori stimati dal sistema: distanza filtrata, scostamento laterale,
    velocita' di avvicinamento, tempo al contatto, livello di allerta;
  - i valori di riferimento: distanza e scostamento laterale ricavati dai
    punti lidar che cadono nel riquadro, velocita' del veicolo dal GPS.

Esecuzione dalla cartella principale del progetto:

    python tools/analizza_sequenza_kitti.py \
        --sequenza ~/Downloads/2011_09_28-2/2011_09_28_drive_0038_sync \
        --calibrazione ~/Downloads/2011_09_28
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.detector import PedestrianDetector
from detection.distance_estimator import DistanceEstimator

PROGETTO = Path(__file__).resolve().parent.parent

# Riquadro ridotto in cui si cercano i punti lidar del pedone: la parte
# centrale, per escludere lo sfondo ai lati e il terreno in basso.
FRAZIONE_LARGHEZZA = 0.5
FRAZIONE_ALTEZZA = (0.15, 0.85)
# Salto di profondita' oltre il quale due punti appartengono a oggetti diversi.
SALTO_GRUPPO_M = 0.5
PUNTI_MINIMI = 3


def leggi_calibrazione(percorso):
    valori = {}
    for riga in Path(percorso).read_text().splitlines():
        if ":" not in riga:
            continue
        chiave, resto = riga.split(":", 1)
        try:
            valori[chiave] = np.array([float(x) for x in resto.split()])
        except ValueError:
            pass
    return valori


def matrice_proiezione(cartella_calib):
    """Matrice che porta i punti del lidar nel sistema rettificato della
    telecamera 2 (coordinate 3D) e nell'immagine (pixel)."""
    c2c = leggi_calibrazione(Path(cartella_calib) / "calib_cam_to_cam.txt")
    v2c = leggi_calibrazione(Path(cartella_calib) / "calib_velo_to_cam.txt")
    velo_cam = np.eye(4)
    velo_cam[:3, :3] = v2c["R"].reshape(3, 3)
    velo_cam[:3, 3] = v2c["T"]
    r_rect = np.eye(4)
    r_rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    p2 = c2c["P_rect_02"].reshape(3, 4)
    # Coordinate rettificate della telecamera 0; la telecamera 2 e' spostata
    # lateralmente di -P2[0,3]/fx.
    velo_rect = r_rect @ velo_cam
    return velo_rect, p2


def leggi_tempi(percorso):
    tempi = []
    for riga in Path(percorso).read_text().splitlines():
        riga = riga.strip()
        if not riga:
            continue
        data, ora = riga.split(" ")
        sec, frazione = ora.split(".")
        t = datetime.strptime(f"{data} {sec}", "%Y-%m-%d %H:%M:%S")
        tempi.append(t.timestamp() + float("0." + frazione))
    return [t - tempi[0] for t in tempi]


def profondita_pedone(box, uv, xyz):
    """Distanza e scostamento laterale dal gruppo di punti piu' vicino."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    a = cx - w * FRAZIONE_LARGHEZZA / 2
    b = cx + w * FRAZIONE_LARGHEZZA / 2
    c = y1 + h * FRAZIONE_ALTEZZA[0]
    d = y1 + h * FRAZIONE_ALTEZZA[1]
    dentro = (uv[:, 0] >= a) & (uv[:, 0] <= b) & (uv[:, 1] >= c) & (uv[:, 1] <= d)
    if dentro.sum() < PUNTI_MINIMI:
        return None, None, int(dentro.sum())
    p = xyz[dentro]
    ordine = np.argsort(p[:, 2])
    p = p[ordine]
    z = p[:, 2]
    # Il pedone e' il primo gruppo compatto con abbastanza punti.
    inizio = 0
    for i in range(1, len(z) + 1):
        if i == len(z) or z[i] - z[i - 1] > SALTO_GRUPPO_M:
            if i - inizio >= PUNTI_MINIMI:
                g = p[inizio:i]
                return float(np.median(g[:, 2])), float(np.median(g[:, 0])), i - inizio
            inizio = i
    return None, None, int(dentro.sum())


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sequenza", required=True)
    pa.add_argument("--calibrazione", required=True)
    pa.add_argument("--focale", choices=["calibrata", "60gradi"], default="calibrata",
                    help="focale reale della telecamera o ipotesi di 60 gradi")
    pa.add_argument("--video", action="store_true", help="salva il video annotato")
    pa.add_argument("--da-video", default=None,
                    help="legge i fotogrammi da un file mp4, con tempo = indice / fps "
                         "come l'interfaccia web, invece che dalle immagini PNG")
    args = pa.parse_args()

    seq = Path(args.sequenza).expanduser()
    nome = seq.name.replace("_sync", "")
    velo_rect, p2 = matrice_proiezione(Path(args.calibrazione).expanduser())
    fx = float(p2[0, 0])

    stimatore = (DistanceEstimator(focal_length_px=fx) if args.focale == "calibrata"
                 else DistanceEstimator())
    detector = PedestrianDetector(confidence_threshold=0.25,
                                  distance_estimator=stimatore)
    detector.audio_alert.play = lambda *a, **k: None
    detector.reset()

    immagini = sorted((seq / "image_02" / "data").glob("*.png"))
    tempi = leggi_tempi(seq / "image_02" / "timestamps.txt")
    cattura = None
    if args.da_video:
        cattura = cv2.VideoCapture(str(Path(args.da_video).expanduser()))
        fps_video = cattura.get(cv2.CAP_PROP_FPS)
        tempi = [k / fps_video for k in range(len(immagini))]
        nome = Path(args.da_video).stem
    uscita = PROGETTO / "results" / "kitti_raw"
    uscita.mkdir(parents=True, exist_ok=True)
    scrittore = None

    righe = []
    for k, img in enumerate(immagini):
        if cattura is not None:
            letto, frame = cattura.read()
            if not letto:
                break
        else:
            frame = cv2.imread(str(img))
        r = detector.process(frame, timestamp=tempi[k])

        velo = np.fromfile(seq / "velodyne_points" / "data" / (img.stem + ".bin"),
                           dtype=np.float32).reshape(-1, 4)
        punti = np.c_[velo[:, :3], np.ones(len(velo))]
        rect = (velo_rect @ punti.T).T[:, :3]
        davanti = rect[:, 2] > 1.0
        rect = rect[davanti]
        uvw = (p2 @ np.c_[rect, np.ones(len(rect))].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        # Coordinate rispetto alla telecamera 2 (stesso asse ottico del frame).
        xyz = rect.copy()
        xyz[:, 0] += p2[0, 3] / fx

        oxts = np.loadtxt(seq / "oxts" / "data" / (img.stem + ".txt"))
        v_veicolo = float(oxts[8])

        for det in r["detections"]:
            box = (det["x1"], det["y1"], det["x2"], det["y2"])
            d_lidar, x_lidar, n = profondita_pedone(box, uv, xyz)
            righe.append({
                "fotogramma": k,
                "tempo_s": round(tempi[k], 3),
                "traccia": det["track_id"],
                "confidenza": round(det["confidence"], 3),
                "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                "troncato": int(det["truncated"]),
                "distanza_grezza_m": round(stimatore.estimate(det["y2"] - det["y1"], frame.shape[1]), 2),
                "distanza_stimata_m": round(det["distance_m"], 2) if det["distance_m"] is not None else "",
                "laterale_stimato_m": round(det["smoothed_lateral_m"], 2) if det["smoothed_lateral_m"] is not None else "",
                "velocita_avvicinamento_mps": round(det["closing_speed_mps"], 2),
                "ttc_s": round(det["ttc_s"], 2) if det["ttc_s"] is not None else "",
                "livello": det["alert_level"],
                "distanza_lidar_m": round(d_lidar, 2) if d_lidar is not None else "",
                "laterale_lidar_m": round(x_lidar, 2) if x_lidar is not None else "",
                "punti_lidar": n,
                "velocita_veicolo_mps": round(v_veicolo, 2),
            })

        if args.video:
            if scrittore is None:
                h, w = r["frame"].shape[:2]
                scrittore = cv2.VideoWriter(str(uscita / f"{nome}_annotato.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            scrittore.write(r["frame"])
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(immagini)}", flush=True)

    if scrittore is not None:
        scrittore.release()
    file_csv = uscita / f"{nome}_{args.focale}.csv"
    with open(file_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"{len(immagini)} fotogrammi, {len(righe)} righe -> {file_csv}")


if __name__ == "__main__":
    main()
