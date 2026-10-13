"""
Confronta la distanza stimata dal modello pinhole con la distanza reale
riportata nelle annotazioni 3D di KITTI.

Esecuzione dalla cartella principale del progetto:

    python3 tools/valuta_distanza_kitti.py
    python3 tools/valuta_distanza_kitti.py --modello runs/detect/runs/kitti_final/weights/best.pt
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.distance_estimator import DistanceEstimator
from detection.model_loader import load_yolo_model

PROGETTO = Path(__file__).resolve().parent.parent
ETICHETTE = (PROGETTO / "dataset" / "kitti" / "raw" / "data_object_label_2"
             / "training" / "label_2")
IOU_MINIMA = 0.5

# Punto principale della telecamera KITTI (matrice P2). Il sistema in
# esercizio non lo conosce e assume il centro dell'immagine: qui serve
# solo per misurare quanto costa quell'approssimazione.
KITTI_CX_PX = 609.5593

# Semilarghezza del corridoio di marcia, in metri: mezza carreggiata
# utile del veicolo piu' un margine.
CORRIDOIO_M = 1.5

# Criteri di visibilita' usati da KITTI per la difficolta' "moderate":
# sotto queste soglie il pedone non e' considerato rilevabile.
ALTEZZA_MINIMA_PX = 25
OCCLUSIONE_MASSIMA = 1
TRONCAMENTO_MASSIMO = 0.30


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def altre_persone(percorso):
    """Box di ciclisti, persone sedute e zone ignorate: non sono errori."""
    out=[]
    for riga in percorso.read_text().splitlines():
        c=riga.split()
        if len(c) >= 15 and c[0] in ("Cyclist", "Person_sitting", "DontCare"):
            out.append((float(c[4]), float(c[5]), float(c[6]), float(c[7])))
    return out


def pedoni_veri(percorso):
    """Righe Pedestrian dell'annotazione: box, altezza reale, distanza reale."""
    out = []
    for riga in percorso.read_text().splitlines():
        c = riga.split()
        if len(c) < 15 or c[0] != "Pedestrian":
            continue
        out.append({
            "box": (float(c[4]), float(c[5]), float(c[6]), float(c[7])),
            "troncamento": float(c[1]),
            "occlusione": int(c[2]),
            "altezza_m": float(c[8]),
            "laterale_m": float(c[11]),
            "distanza_m": float(c[13]),
        })
    for g in out:
        altezza_px = g["box"][3] - g["box"][1]
        g["visibile"] = (
            altezza_px >= ALTEZZA_MINIMA_PX
            and g["occlusione"] <= OCCLUSIONE_MASSIMA
            and g["troncamento"] <= TRONCAMENTO_MASSIMO
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--immagini", type=int, default=300)
    p.add_argument("--insieme", default="val", choices=["val", "train"])
    p.add_argument("--confidenza", type=float, default=0.4)
    p.add_argument("--etichette", default=str(ETICHETTE))
    p.add_argument("--modello", default=None,
                   help="pesi da usare; senza questo argomento yolov8n.pt")
    p.add_argument("--uscita", default="distanza_kitti.csv",
                   help="nome del CSV di dettaglio, salvato in results/")
    args = p.parse_args()

    cartella_img = PROGETTO / "dataset" / "kitti" / "images" / args.insieme
    cartella_lab = Path(args.etichette).expanduser()
    if not cartella_lab.is_dir():
        sys.exit(f"Annotazioni non trovate in {cartella_lab}")

    immagini = sorted(cartella_img.glob("*.png"))[: args.immagini]
    if not immagini:
        sys.exit(f"Nessuna immagine in {cartella_img}")

    pesi = Path(args.modello).expanduser() if args.modello else None
    if pesi and not pesi.is_absolute():
        pesi = PROGETTO / pesi
    modello = load_yolo_model(str(pesi) if pesi else None)
    print(f"modello: {pesi.name if pesi else 'yolov8n.pt'}")
    stimatore = DistanceEstimator.for_kitti()

    # Fasce di distanza reale usate per il richiamo e per l'errore.
    fasce = [(0, 10), (10, 20), (20, 30), (30, 1000)]
    visibili_per_fascia = [0] * len(fasce)
    trovati_per_fascia = [0] * len(fasce)

    righe = []
    n_veri = n_visibili = n_rilevati = n_accoppiati_visibili = 0
    n_su_altre_persone = n_spuri = 0

    for k, img in enumerate(immagini, 1):
        lab = cartella_lab / (img.stem + ".txt")
        if not lab.exists():
            continue
        veri = pedoni_veri(lab)
        n_veri += len(veri)
        n_visibili += sum(1 for g in veri if g["visibile"])

        r = modello.predict(source=str(img), classes=[0],
                            conf=args.confidenza, verbose=False)[0]
        larghezza = r.orig_shape[1]
        rilevati = []
        if r.boxes is not None and len(r.boxes):
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy()):
                rilevati.append((tuple(float(v) for v in box), float(conf)))
        rilevati.sort(key=lambda t: -t[1])
        n_rilevati += len(rilevati)

        altri = altre_persone(lab)
        liberi = list(range(len(veri)))
        for box, conf in rilevati:
            migliore, valore = None, IOU_MINIMA
            for i in liberi:
                v = iou(box, veri[i]["box"])
                if v >= valore:
                    migliore, valore = i, v
            if migliore is None:
                # Non e' un pedone annotato: distinguo fra ciclista o
                # persona seduta e rilevamento davvero spurio.
                if any(iou(box, b) >= IOU_MINIMA for b in altri):
                    n_su_altre_persone += 1
                else:
                    n_spuri += 1
                continue
            liberi.remove(migliore)
            g = veri[migliore]
            if g["visibile"]:
                n_accoppiati_visibili += 1

            h_px = box[3] - box[1]
            x_centro = (box[0] + box[2]) / 2.0
            # Scostamento laterale: la focale si semplifica, resta solo
            # l'ipotesi sull'altezza del pedone.
            lat_stimato = (x_centro - larghezza / 2.0) * 1.70 / h_px
            lat_con_cx = (x_centro - KITTI_CX_PX) * 1.70 / h_px
            lat_reale = g["laterale_m"]
            stimata = stimatore.estimate(h_px, larghezza)
            if stimata is None:
                continue
            reale = g["distanza_m"]
            con_altezza_vera = (
                DistanceEstimator.for_kitti(pedestrian_height_m=g["altezza_m"])
                .estimate(h_px, larghezza)
            )
            righe.append({
                "immagine": img.stem,
                "iou": round(valore, 3),
                "confidenza": round(conf, 3),
                "altezza_box_px": round(h_px, 1),
                "distanza_stimata_m": round(stimata, 2),
                "distanza_reale_m": round(reale, 2),
                "errore_m": round(stimata - reale, 2),
                "errore_pct": round(100 * (stimata - reale) / reale, 1),
                "stima_con_altezza_vera_m": round(con_altezza_vera, 2)
                if con_altezza_vera else "",
                "laterale_stimato_m": round(lat_stimato, 2),
                "laterale_reale_m": round(lat_reale, 2),
                "errore_laterale_m": round(lat_stimato - lat_reale, 2),
                "laterale_con_cx_noto_m": round(lat_con_cx, 2),
                "nel_corridoio_stimato": int(abs(lat_stimato) <= CORRIDOIO_M),
                "nel_corridoio_reale": int(abs(lat_reale) <= CORRIDOIO_M),
                "altezza_reale_m": g["altezza_m"],
                "troncamento": g["troncamento"],
                "occlusione": g["occlusione"],
                "visibile": int(g["visibile"]),
            })

        # Richiamo per fascia: pedoni visibili trovati, divisi per distanza.
        for i, g in enumerate(veri):
            if not g["visibile"]:
                continue
            for j, (a, b) in enumerate(fasce):
                if a <= g["distanza_m"] < b:
                    visibili_per_fascia[j] += 1
                    if i not in liberi:
                        trovati_per_fascia[j] += 1
                    break

        if k % 50 == 0:
            print(f"  {k}/{len(immagini)} immagini")

    if not righe:
        sys.exit("Nessun accoppiamento trovato.")

    uscita = PROGETTO / "results" / args.uscita
    uscita.parent.mkdir(exist_ok=True)
    with uscita.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    err = [r["errore_m"] for r in righe]
    pct = [r["errore_pct"] for r in righe]
    ass = [abs(e) for e in err]
    alt = [r["altezza_reale_m"] for r in righe]

    print()
    print("=" * 58)
    print(f"  immagini analizzate      {len(immagini)}")
    print(f"  pedoni annotati          {n_veri}")
    print(f"  di cui visibili          {n_visibili}")
    print(f"  rilevamenti prodotti     {n_rilevati}")
    print(f"  accoppiati (IoU>={IOU_MINIMA})   {len(righe)}")
    if n_visibili:
        print(f"  quota di visibili trovati {100*n_accoppiati_visibili/n_visibili:.1f} %")
    print(f"  su ciclisti o persone sedute {n_su_altre_persone}")
    print(f"  spuri                    {n_spuri}")
    if n_rilevati:
        print(f"  precisione grezza        {100*len(righe)/n_rilevati:.1f} %")
        print(f"  precisione al netto      "
              f"{100*len(righe)/(n_rilevati - n_su_altre_persone):.1f} %")
    print("-" * 58)
    print("  pedoni visibili trovati, per fascia di distanza reale:")
    for (a, b), nv, nt in zip(fasce, visibili_per_fascia, trovati_per_fascia):
        eti = f"{a}-{b} m" if b < 1000 else f"oltre {a} m"
        quota = f"{100*nt/nv:5.1f} %" if nv else "   -"
        print(f"    {eti:>10}  visibili {nv:>4}  trovati {nt:>4}  {quota}")
    print("-" * 58)
    print(f"  errore medio             {statistics.mean(err):+.2f} m")
    print(f"  errore assoluto medio    {statistics.mean(ass):.2f} m")
    print(f"  errore assoluto mediano  {statistics.median(ass):.2f} m")
    print(f"  errore percentuale medio {statistics.mean(pct):+.1f} %")
    print(f"  altezza reale media      {statistics.mean(alt):.2f} m "
          f"(il modello ne assume 1,70)")
    print("-" * 58)
    print("  per fascia di distanza reale:")
    for a, b in fasce:
        s = [r for r in righe if a <= r["distanza_reale_m"] < b]
        if not s:
            continue
        m = statistics.mean(abs(r["errore_m"]) for r in s)
        mp = statistics.mean(r["errore_pct"] for r in s)
        eti = f"{a}-{b} m" if b < 1000 else f"oltre {a} m"
        print(f"    {eti:>10}  n={len(s):>4}  err.ass. {m:5.2f} m   err. {mp:+6.1f} %")
    print("-" * 58)
    with_vera = [r for r in righe if r["stima_con_altezza_vera_m"] != ""]
    if with_vera:
        e2 = [abs(r["stima_con_altezza_vera_m"] - r["distanza_reale_m"])
              for r in with_vera]
        print(f"  usando l'altezza reale di ciascun pedone al posto di 1,70 m:")
        print(f"    errore assoluto medio  {statistics.mean(e2):.2f} m")
    print("-" * 58)
    el = [abs(r["errore_laterale_m"]) for r in righe]
    e2l = [abs(r["laterale_con_cx_noto_m"] - r["laterale_reale_m"]) for r in righe]
    print("  scostamento laterale:")
    print(f"    errore assoluto medio    {statistics.mean(el):.2f} m")
    print(f"    errore assoluto mediano  {statistics.median(el):.2f} m")
    print(f"    con il punto principale noto {statistics.mean(e2l):.2f} m")
    vp = sum(1 for r in righe if r["nel_corridoio_stimato"] and r["nel_corridoio_reale"])
    vn = sum(1 for r in righe if not r["nel_corridoio_stimato"] and not r["nel_corridoio_reale"])
    fp = sum(1 for r in righe if r["nel_corridoio_stimato"] and not r["nel_corridoio_reale"])
    fn = sum(1 for r in righe if not r["nel_corridoio_stimato"] and r["nel_corridoio_reale"])
    print(f"  decisione corridoio (|X| <= {CORRIDOIO_M} m):")
    print(f"    dentro, giudicati dentro   {vp}")
    print(f"    fuori,  giudicati fuori    {vn}")
    print(f"    fuori,  giudicati dentro   {fp}   (allarme non necessario)")
    print(f"    dentro, giudicati fuori    {fn}   (pedone declassato a torto)")
    print(f"    decisioni corrette         {100*(vp+vn)/len(righe):.1f} %")
    print("=" * 58)
    print(f"\nDettaglio per pedone in {uscita.relative_to(PROGETTO)}")


if __name__ == "__main__":
    main()
