"""
Valutazione del sistema sul dataset iseAuto (versione 2).

Il dataset contiene, per quattro condizioni (giorno sereno, giorno con
pioggia, notte serena, notte con pioggia), 600 fotogrammi annotati a mano
con maschere di segmentazione (0 sfondo, 1 veicolo, 2 persona) e i punti
lidar proiettati sull'immagine. Da questi dati lo script ricava:

  - i riquadri delle persone, come componenti connesse della maschera;
  - la distanza reale di ciascuna persona, come mediana della profondita'
    dei punti lidar che cadono sulla sua sagoma;
  - la lunghezza focale della telecamera, stimata dalle coppie
    punto 3D / pixel (il dataset non include un file di calibrazione).

Esecuzione dalla cartella principale del progetto:

    python tools/valuta_iseauto.py
    python tools/valuta_iseauto.py --modello runs/detect/runs/kitti_final/weights/best.pt
"""

import argparse
import csv
import math
import pickle
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.model_loader import load_yolo_model

PROGETTO = Path(__file__).resolve().parent.parent
RADICE = PROGETTO / "dataset" / "iseauto"
CONDIZIONI = ["day_fair", "day_rain", "night_fair", "night_rain"]
NOMI = {"day_fair": "giorno sereno", "day_rain": "giorno pioggia",
        "night_fair": "notte serena", "night_rain": "notte pioggia"}

ALTEZZA_PEDONE_M = 1.70
IOU_MINIMA = 0.5
CLASSE_PERSONA = 2
# Altezza minima perche' una persona sia considerata rilevabile. Equivale,
# nell'immagine ridotta a 640 px che il modello analizza, ai 25 px del
# criterio KITTI: 25 * (1242/640) * (640/4240) ~ 85 px sull'immagine piena.
ALTEZZA_MINIMA_PX = 85
PUNTI_LIDAR_MINIMI = 3
# Salto di profondita' oltre il quale due punti appartengono a oggetti diversi.
SALTO_GRUPPO_M = 0.5
FASCE = [(0, 10), (10, 20), (20, 30), (30, 1000)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def carica_lidar(percorso):
    d = pickle.load(open(percorso, "rb"))
    c = d["camera_coordinates"].astype(float)
    m = c[:, 0] == 1  # punti che cadono nell'immagine
    return c[m, 1:3], d["3d_points"][m].astype(float)


def stima_proiezione(file_pkl, punti_per_file=2000):
    """Matrice di proiezione 3x4 (DLT) dalle coppie punto 3D / pixel."""
    rng = np.random.default_rng(0)
    uv_tot, xyz_tot = [], []
    for f in file_pkl:
        uv, xyz = carica_lidar(f)
        k = rng.choice(len(xyz), min(punti_per_file, len(xyz)), replace=False)
        uv_tot.append(uv[k]); xyz_tot.append(xyz[k])
    uv = np.vstack(uv_tot); xyz = np.vstack(xyz_tot)
    n = len(xyz)
    xh = np.c_[xyz, np.ones(n)]
    a = np.zeros((2 * n, 12))
    a[0::2, 0:4] = xh; a[0::2, 8:12] = -uv[:, [0]] * xh
    a[1::2, 4:8] = xh; a[1::2, 8:12] = -uv[:, [1]] * xh
    p = np.linalg.svd(a, full_matrices=False)[2][-1].reshape(3, 4)
    # Terza riga normalizzata: il prodotto con un punto da' la profondita'
    # lungo l'asse ottico, in metri.
    p = p / np.linalg.norm(p[2, :3])
    if (xh @ p[2]).mean() < 0:
        p = -p
    pr = xh @ p.T
    errore = np.mean(np.hypot(pr[:, 0] / pr[:, 2] - uv[:, 0],
                              pr[:, 1] / pr[:, 2] - uv[:, 1]))
    # Parametri interni con la scomposizione RQ della parte 3x3.
    m = p[:, :3]
    rev = np.flipud(np.eye(3))
    q, r = np.linalg.qr((rev @ m).T)
    k = rev @ r.T @ rev
    k = k @ np.diag(np.sign(np.diag(k)))
    k = k / k[2, 2]
    return p, k, errore


def persone_annotate(percorso_maschera, larghezza, altezza):
    """Riquadri delle persone (immagine piena) ed etichette per pixel."""
    g = cv2.imread(str(percorso_maschera), cv2.IMREAD_GRAYSCALE)
    sx, sy = larghezza / g.shape[1], altezza / g.shape[0]
    maschera = (g == CLASSE_PERSONA).astype(np.uint8)
    n, etichette, stat, _ = cv2.connectedComponentsWithStats(maschera, 8)
    persone = []
    for i in range(1, n):
        x, y, w, h, area = stat[i]
        persone.append({
            "id": i,
            "box": (x * sx, y * sy, (x + w) * sx, (y + h) * sy),
            "altezza_px": h * sy,
        })
        persone[-1]["visibile"] = persone[-1]["altezza_px"] >= ALTEZZA_MINIMA_PX
    return persone, etichette, sx, sy


def profondita_persone(persone, etichette, sx, sy, uv, xyz, p):
    """Mediana della profondita' dei punti lidar sulla sagoma di ciascuno."""
    if not persone:
        return
    col = np.clip((uv[:, 0] / sx).astype(int), 0, etichette.shape[1] - 1)
    rig = np.clip((uv[:, 1] / sy).astype(int), 0, etichette.shape[0] - 1)
    lab = etichette[rig, col]
    z = np.c_[xyz, np.ones(len(xyz))] @ p[2]
    for g in persone:
        zz = np.sort(z[lab == g["id"]])
        g["punti_lidar"] = int(len(zz))
        g["distanza_m"] = None
        if len(zz) < PUNTI_LIDAR_MINIMI:
            continue
        # La sagoma annotata e' approssimata e il lidar e' rado: fra i punti
        # che vi cadono ci sono anche quelli dello sfondo, piu' lontani.
        # I punti vengono divisi in gruppi separati da salti di profondita'
        # e si prende il gruppo piu' numeroso, che corrisponde alla persona.
        tagli = np.where(np.diff(zz) > SALTO_GRUPPO_M)[0] + 1
        gruppi = np.split(zz, tagli)
        principale = max(gruppi, key=len)
        if len(principale) >= PUNTI_LIDAR_MINIMI:
            g["distanza_m"] = float(np.median(principale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radice", default=str(RADICE))
    ap.add_argument("--modello", default=None)
    ap.add_argument("--confidenza", type=float, default=0.25)
    ap.add_argument("--limite", type=int, default=0,
                    help="fotogrammi per condizione (0 = tutti)")
    ap.add_argument("--uscita", default="iseauto_generico.csv")
    args = ap.parse_args()

    radice = Path(args.radice).expanduser() / "labeled"
    if not radice.is_dir():
        sys.exit(f"Cartella non trovata: {radice}")

    campione = sorted((radice / "day_fair" / "pkl").glob("*.pkl"))[:20]
    p, k, errore = stima_proiezione(campione)
    fx = k[0, 0]
    print(f"calibrazione stimata: fx {fx:.1f} px, fy {k[1,1]:.1f} px, "
          f"cx {k[0,2]:.1f}, cy {k[1,2]:.1f}, errore di riproiezione {errore:.2f} px")

    pesi = Path(args.modello).expanduser() if args.modello else None
    if pesi and not pesi.is_absolute():
        pesi = PROGETTO / pesi
    modello = load_yolo_model(str(pesi) if pesi else None)
    print(f"modello: {pesi.name if pesi else 'yolov8n.pt'}")

    righe = []
    sintesi = {}
    for cond in CONDIZIONI:
        immagini = sorted((radice / cond / "rgb").glob("*.png"))
        if args.limite:
            immagini = immagini[: args.limite]
        s = dict(visibili=0, trovati=0, rilevamenti=0, spuri=0,
                 vis_fascia=[0]*len(FASCE), trov_fascia=[0]*len(FASCE))
        for n_img, img in enumerate(immagini, 1):
            mask = radice / cond / "annotation_gray" / img.name
            pkl = radice / cond / "pkl" / (img.stem + ".pkl")
            if not mask.exists() or not pkl.exists():
                continue
            r = modello.predict(source=str(img), classes=[0],
                                conf=args.confidenza, verbose=False)[0]
            alt, larg = r.orig_shape
            persone, etichette, sx, sy = persone_annotate(mask, larg, alt)
            uv, xyz = carica_lidar(pkl)
            profondita_persone(persone, etichette, sx, sy, uv, xyz, p)

            rilevati = []
            if r.boxes is not None and len(r.boxes):
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    rilevati.append((tuple(float(v) for v in b), float(c)))
            rilevati.sort(key=lambda t: -t[1])
            s["rilevamenti"] += len(rilevati)

            liberi = list(range(len(persone)))
            for box, conf in rilevati:
                migliore, valore = None, IOU_MINIMA
                for i in liberi:
                    v = iou(box, persone[i]["box"])
                    if v >= valore:
                        migliore, valore = i, v
                if migliore is None:
                    s["spuri"] += 1
                    continue
                liberi.remove(migliore)
                g = persone[migliore]
                if g["distanza_m"] is None:
                    continue
                h = box[3] - box[1]
                stimata = fx * ALTEZZA_PEDONE_M / h
                f60 = (larg / 2) / math.tan(math.radians(30))
                stimata60 = f60 * ALTEZZA_PEDONE_M / h
                reale = g["distanza_m"]
                righe.append({
                    "condizione": cond, "immagine": img.stem,
                    "iou": round(valore, 3), "confidenza": round(conf, 3),
                    "altezza_box_px": round(h, 1),
                    "distanza_reale_m": round(reale, 2),
                    "punti_lidar": g["punti_lidar"],
                    "distanza_stimata_m": round(stimata, 2),
                    "errore_m": round(stimata - reale, 2),
                    "errore_pct": round(100 * (stimata - reale) / reale, 1),
                    "stimata_60gradi_m": round(stimata60, 2),
                    "errore_60gradi_pct": round(100 * (stimata60 - reale) / reale, 1),
                    "visibile": int(g["visibile"]),
                })

            for i, g in enumerate(persone):
                if not g["visibile"]:
                    continue
                s["visibili"] += 1
                trovato = i not in liberi
                s["trovati"] += trovato
                d = g["distanza_m"]
                if d is not None:
                    for j, (a, b) in enumerate(FASCE):
                        if a <= d < b:
                            s["vis_fascia"][j] += 1
                            s["trov_fascia"][j] += trovato
                            break
            if n_img % 100 == 0:
                print(f"  {NOMI[cond]}: {n_img}/{len(immagini)}")
        sintesi[cond] = s

    uscita = PROGETTO / "results" / args.uscita
    if righe:
        with uscita.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
            w.writeheader(); w.writerows(righe)

    print()
    print("=" * 74)
    print(f"{'condizione':<16}{'visibili':>9}{'trovati':>9}{'richiamo':>10}"
          f"{'rilev.':>8}{'spuri':>7}{'precisione':>12}")
    for cond, s in sintesi.items():
        ric = 100 * s["trovati"] / s["visibili"] if s["visibili"] else float("nan")
        prec = 100 * (s["rilevamenti"] - s["spuri"]) / s["rilevamenti"] if s["rilevamenti"] else float("nan")
        print(f"{NOMI[cond]:<16}{s['visibili']:>9}{s['trovati']:>9}{ric:>9.1f}%"
              f"{s['rilevamenti']:>8}{s['spuri']:>7}{prec:>11.1f}%")
    print("-" * 74)
    print("richiamo per fascia di distanza (persone con distanza lidar):")
    for cond, s in sintesi.items():
        celle = []
        for (a, b), nv, nt in zip(FASCE, s["vis_fascia"], s["trov_fascia"]):
            eti = f"{a}-{b}" if b < 1000 else f">{a}"
            celle.append(f"{eti} m {100*nt/nv:5.1f}% ({nv})" if nv else f"{eti} m    -  ")
        print(f"  {NOMI[cond]:<15} " + "  ".join(celle))
    print("-" * 74)
    print("distanza (persone trovate, con distanza lidar):")
    for cond in CONDIZIONI:
        rr = [r for r in righe if r["condizione"] == cond]
        if not rr:
            continue
        ass = [abs(r["errore_m"]) for r in rr]
        print(f"  {NOMI[cond]:<15} n={len(rr):>4}  err.ass.medio {statistics.mean(ass):5.2f} m"
              f"  mediano {statistics.median(ass):5.2f} m"
              f"  con segno {statistics.mean(r['errore_pct'] for r in rr):+5.1f}%"
              f"  | con 60 gradi {statistics.mean(r['errore_60gradi_pct'] for r in rr):+5.1f}%")
    print("=" * 74)
    if righe:
        print(f"\nDettaglio in {uscita.relative_to(PROGETTO)}")


if __name__ == "__main__":
    main()
