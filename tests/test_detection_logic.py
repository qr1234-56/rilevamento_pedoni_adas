"""
Verifica della logica di stima della distanza, di allerta, di stato delle
tracce e di soppressione dei duplicati.

Questi moduli non dipendono da YOLO ne' da OpenCV, quindi il test gira in
pochi decimi di secondo senza caricare il modello.

Esecuzione dalla cartella principale del progetto:

    python3 tests/test_detection_logic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.distance_estimator import (
    DistanceEstimator,
    focal_length_from_fov,
    is_truncated,
    KITTI_FOCAL_LENGTH_PX,
)
from detection.alert_manager import AlertManager
from detection.track_state import TrackStateStore

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  OK   {name}")
    else:
        fail += 1; print(f"  FAIL {name}  {detail}")

print("═══ 1. Modello pinhole ═══")
kitti = DistanceEstimator.for_kitti()
# Un pedone di 1,70 m a 10 m in KITTI occupa f*H/d pixel di altezza.
h = KITTI_FOCAL_LENGTH_PX * 1.70 / 10.0
d = kitti.estimate(h, 1242)
check("round-trip KITTI a 10 m", abs(d - 10.0) < 0.01, f"ottenuto {d}")

for true_d in (5, 15, 30, 60):
    h = KITTI_FOCAL_LENGTH_PX * 1.70 / true_d
    got = kitti.estimate(h, 1242)
    check(f"round-trip a {true_d} m", abs(got - true_d) < 0.01, f"ottenuto {got}")

f640 = focal_length_from_fov(640, 60.0)
check("focale da FOV 60° su 640px", abs(f640 - 554.256) < 0.01, f"ottenuto {f640}")

check("altezza box nulla -> None", kitti.estimate(0, 1242) is None)
check("saturazione minima", kitti.estimate(100000, 1242) >= 0.5)
check("saturazione massima", kitti.estimate(1, 1242) <= 120.0)

check("box troncato in basso", is_truncated(100, 375, 375) is True)
check("box troncato in alto", is_truncated(1, 200, 375) is True)
check("box interno", is_truncated(50, 300, 375) is False)

print("\n═══ 2. Il criterio basato sul rapporto fra altezze ═══")
# Stesso pedone, stessa distanza reale, due sorgenti diverse.
TRUE_D = 15.0
# KITTI: 1242x375, focale nota
h_kitti = KITTI_FOCAL_LENGTH_PX * 1.70 / TRUE_D
ratio_kitti = h_kitti / 375
# Video da telefono 1920x1080, FOV 60°
f_phone = focal_length_from_fov(1920, 60.0)
h_phone = f_phone * 1.70 / TRUE_D
ratio_phone = h_phone / 1080

phone = DistanceEstimator(hfov_deg=60.0)
d_kitti = kitti.estimate(h_kitti, 1242)
d_phone = phone.estimate(h_phone, 1920)

print(f"  indice vecchio: KITTI {ratio_kitti:.3f}  vs  telefono {ratio_phone:.3f}"
      f"   -> scarto {abs(ratio_kitti-ratio_phone)/ratio_phone*100:.0f}%")
print(f"  distanza nuova: KITTI {d_kitti:.1f} m  vs  telefono {d_phone:.1f} m")
check("il vecchio indice differisce fra sorgenti",
      abs(ratio_kitti - ratio_phone) / ratio_phone > 0.20)
check("la nuova stima coincide", abs(d_kitti - d_phone) < 0.01)

print("\n═══ 3. Livelli di allerta ═══")
am = AlertManager()
check("40 m -> verde", am.evaluate(1, 40.0, 0.9) == "green")
check("20 m -> giallo", am.evaluate(2, 20.0, 0.9) == "yellow")
check("8 m -> rosso", am.evaluate(3, 8.0, 0.9) == "red")

print("\n  isteresi (il rosso non si spegne appena superata la soglia)")
am2 = AlertManager()
am2.evaluate(1, 8.0, 0.9)                       # entra in rosso
l = am2.evaluate(1, 11.0, 0.9)                  # oltre i 10 m ma dentro l'isteresi
check("11 m resta rosso (soglia 10 + 2,5)", l == "red", f"ottenuto {l}")
for _ in range(5):
    l = am2.evaluate(1, 14.0, 0.9)              # oltre 12,5 m: declassa
check("14 m declassa a giallo dopo conferme", l == "yellow", f"ottenuto {l}")

print("\n  conferma temporale")
am3 = AlertManager(frames_to_relax=5)
am3.evaluate(1, 8.0, 0.9)
l1 = am3.evaluate(1, 40.0, 0.9)                 # un solo frame lontano
check("un frame isolato non spegne il rosso", l1 == "red", f"ottenuto {l1}")
for _ in range(4):
    l2 = am3.evaluate(1, 40.0, 0.9)
check("cinque frame consecutivi lo spengono", l2 == "green", f"ottenuto {l2}")

print("\n  soglie di confidenza")
am4 = AlertManager()
check("rosso con confidenza 0.40 (>0.35) resta rosso",
      am4.evaluate(1, 5.0, 0.40) == "red")
am5 = AlertManager()
check("rosso con confidenza 0.20 declassa",
      am5.evaluate(1, 5.0, 0.20) != "red")
am6 = AlertManager()
check("giallo con confidenza 0.45 (<0.50) declassa a verde",
      am6.evaluate(1, 20.0, 0.45) == "green")

print("\n  time-to-collision")
am7 = AlertManager()
# Lontano ma in rapido avvicinamento: il rosso richiede conferma su piu'
# fotogrammi, perche' il tempo al contatto e' una grandezza rumorosa.
for _ in range(3):
    l = am7.evaluate(1, 30.0, 0.9, ttc_s=1.5)
check("30 m con TTC 1,5 s confermato -> rosso", l == "red", f"ottenuto {l}")

am7b = AlertManager()
am7b.evaluate(1, 30.0, 0.9, ttc_s=None)
l = am7b.evaluate(1, 30.0, 0.9, ttc_s=1.5)      # valore isolato, verosimilmente rumore
check("un solo fotogramma con TTC basso non fa scattare il rosso",
      l == "green", f"ottenuto {l}")

am7c = AlertManager()
l = am7c.evaluate(1, 8.0, 0.9)                  # vicinanza: nessuna attesa
check("la vicinanza fa scattare il rosso al primo fotogramma",
      l == "red", f"ottenuto {l}")
am8 = AlertManager()
check("30 m senza avvicinamento -> verde",
      am8.evaluate(1, 30.0, 0.9, ttc_s=None) == "green")

print("\n  box troncato")
am9 = AlertManager()
am9.evaluate(1, 8.0, 0.9)
for _ in range(6):
    l = am9.evaluate(1, 40.0, 0.9, truncated=True)
check("un box troncato non declassa il livello", l == "red", f"ottenuto {l}")

check("most_severe", AlertManager.most_severe(["green", "red", "yellow"]) == "red")
check("most_severe lista vuota", AlertManager.most_severe([]) == "green")

print("\n  pulizia dello stato")
am10 = AlertManager()
am10.evaluate(1, 5.0, 0.9); am10.evaluate(2, 5.0, 0.9)
am10.prune([1])
check("prune elimina le tracce assenti", len(am10._states) == 1)
am10.reset()
check("reset azzera tutto", len(am10._states) == 0)

print("\n═══ 4. Stato delle tracce ═══")
ts = TrackStateStore()
# Pedone che si avvicina: 20 m -> 19 m in 0,2 s = 5 m/s di avvicinamento.
ts.update(1, (0, 0, 50, 150), 20.0, 0.0)
for i in range(1, 12):
    r = ts.update(1, (0, 0, 50, 150 + i), 20.0 - i * 1.0, i * 0.2)
check("avvicinamento rilevato", r["depth_direction"] == "approaching",
      f"ottenuto {r['depth_direction']}")
check("velocita' di avvicinamento positiva", r["closing_speed_mps"] > 0)
check("TTC calcolato", r["ttc_s"] is not None and r["ttc_s"] > 0,
      f"ttc={r['ttc_s']}")
print(f"  distanza filtrata {r['smoothed_distance_m']:.1f} m, "
      f"avvicinamento {r['closing_speed_mps']:.2f} m/s, TTC {r['ttc_s']:.1f} s")

ts2 = TrackStateStore()
ts2.update(2, (0, 0, 50, 150), 10.0, 0.0)
for i in range(1, 6):
    r2 = ts2.update(2, (0, 0, 50, 150), 10.0 + i * 1.0, i * 0.2)
check("allontanamento rilevato", r2["depth_direction"] == "receding",
      f"ottenuto {r2['depth_direction']}")
check("nessun TTC in allontanamento", r2["ttc_s"] is None)

# Cambio di postura: altezza cresce del 30% ma il bordo inferiore non si muove.
ts3 = TrackStateStore()
ts3.update(3, (0, 100, 50, 200), 10.0, 0.0)     # altezza 100, bordo a 200
r3 = ts3.update(3, (0, 70, 50, 201), 7.5, 0.2)  # altezza 131, bordo fermo
check("cambio di postura riconosciuto", r3["posture_change"] is True)
check("postura non produce TTC", r3["ttc_s"] is None)
check("postura non produce avvicinamento", r3["depth_direction"] == "stable")

ts4 = TrackStateStore()
first = ts4.update(4, (0, 0, 10, 10), 5.0, 0.0)
check("prima osservazione senza TTC", first["ttc_s"] is None)
check("prima osservazione senza movimento", first["movement_distance"] == 0.0)

ts4.update(5, (0, 0, 10, 10), 5.0, 0.0)
ts4.prune([4])
check("prune sullo stato tracce", len(ts4._states) == 1)

def scenario(speed_kmh, start_m=45.0, stop_m=4.0, fps=10.0, conf=0.85):
    """Simula un avvicinamento a velocita' costante e restituisce i livelli."""
    am = AlertManager()
    ts = TrackStateStore()
    est = DistanceEstimator.for_kitti()
    speed = speed_kmh / 3.6
    step = speed / fps
    seq = []
    d_true = start_m
    i = 0
    while d_true > stop_m and i < 2000:
        h_px = KITTI_FOCAL_LENGTH_PX * 1.70 / d_true
        d = est.estimate(h_px, 1242)
        st = ts.update(1, (0, 0, 40, int(h_px)), d, i / fps)
        lv = am.evaluate(1, st["smoothed_distance_m"], conf, st["ttc_s"])
        seq.append((d_true, lv))
        d_true -= step
        i += 1
    return seq

print("\n═══ 5. Avvicinamento urbano a 30 km/h ═══")
seq = scenario(30.0)
levels = [l for _, l in seq]
transitions = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
first = {lv: next(d for d, l in seq if l == lv) for lv in set(levels)}
print("  " + "  ".join(f"{d:.0f}m:{l[0].upper()}" for d, l in seq[::6]))
print(f"  primo giallo a {first.get('yellow', 0):.1f} m, "
      f"primo rosso a {first.get('red', 0):.1f} m")
check("parte dal verde", levels[0] == "green")
check("attraversa il giallo", "yellow" in levels)
check("finisce in rosso", levels[-1] == "red")
# A 30 km/h (8,33 m/s) un TTC di 4 s corrisponde a circa 33 m: la regola sul
# tempo residuo anticipa quindi la soglia di distanza dei 25 m, ed e' il
# comportamento voluto — la distanza fa da minimo garantito, il TTC da
# indicatore anticipato. Il giallo deve comparire fra le due grandezze.
ttc_yellow_distance = (30.0 / 3.6) * AlertManager().yellow_ttc_s
print(f"  (soglia distanza 25,0 m; soglia TTC equivalente "
      f"{ttc_yellow_distance:.1f} m -> comanda il TTC)")
check("giallo governato dal TTC, non oltre la sua soglia",
      25.0 <= first.get("yellow", 0) <= ttc_yellow_distance + 1.0,
      f"primo giallo a {first.get('yellow')}")
check("esattamente due transizioni", transitions == 2,
      f"transizioni={transitions}")

print("\n═══ 6. Avvicinamento veloce: il TTC anticipa l'allerta ═══")
seq_fast = scenario(60.0)
levels_fast = [l for _, l in seq_fast]
first_red_fast = next(d for d, l in seq_fast if l == "red")
first_red_slow = first.get("red", 0)
print(f"  primo rosso a 60 km/h: {first_red_fast:.1f} m  "
      f"(a 30 km/h era {first_red_slow:.1f} m)")
check("a velocita' doppia il rosso scatta prima",
      first_red_fast > first_red_slow,
      f"{first_red_fast} vs {first_red_slow}")
check("nessun lampeggio anche ad alta velocita'",
      sum(1 for a, b in zip(levels_fast, levels_fast[1:]) if a != b) <= 3)

print("\n═══ 7. Pedone fermo a bordo strada (non deve allarmare) ═══")
am_s = AlertManager()
ts_s = TrackStateStore()
est_s = DistanceEstimator.for_kitti()
h_static = KITTI_FOCAL_LENGTH_PX * 1.70 / 30.0
lv_static = []
for i in range(40):
    # Piccolo rumore sull'altezza del box, come accade nella pratica.
    h_noisy = h_static * (1.0 + (0.02 if i % 2 else -0.02))
    d = est_s.estimate(h_noisy, 1242)
    st = ts_s.update(1, (0, 0, 40, int(h_noisy)), d, i * 0.1)
    lv_static.append(
        am_s.evaluate(1, st["smoothed_distance_m"], 0.85, st["ttc_s"])
    )
check("un pedone fermo a 30 m resta verde",
      set(lv_static) == {"green"}, f"livelli osservati {set(lv_static)}")
check("il rumore sul box non genera transizioni",
      sum(1 for a, b in zip(lv_static, lv_static[1:]) if a != b) == 0)


print("\n\u2550\u2550\u2550 9. Allontanamento con riquadro troncato \u2550\u2550\u2550")
# Regressione: un riquadro troncato non deve impedire per sempre
# l'abbassamento del livello. Il riquadro di un pedone che attraversa in
# primo piano tocca il bordo dell'immagine per tutta la durata del
# passaggio, e senza limiti il livello resterebbe ROSSO anche mentre la
# persona si allontana.

def recede(truncated, use_receding=True, start=6.0, end=40.0,
           speed=1.4, fps=5.0, conf=0.85):
    am = AlertManager()
    ts = TrackStateStore()
    est = DistanceEstimator.for_kitti()
    step = speed / fps
    distance = start
    i = 0
    seq = []
    while distance < end and i < 600:
        h = KITTI_FOCAL_LENGTH_PX * 1.70 / distance
        st = ts.update(1, (0, 0, 40, int(h)), est.estimate(h, 1242), i / fps)
        rec = (st["depth_direction"] == "receding") if use_receding else False
        seq.append(
            (distance, am.evaluate(1, st["smoothed_distance_m"], conf,
                                   st["ttc_s"], truncated, rec))
        )
        distance += step
        i += 1
    return seq

seq = recede(truncated=True)
levels = [lv for _, lv in seq]
check("troncato + allontanamento: esce dal rosso", levels[-1] != "red",
      f"livello finale {levels[-1]}")
check("troncato + allontanamento: arriva al verde", levels[-1] == "green",
      f"livello finale {levels[-1]}")

# Anche se il tracker non riconosce l'allontanamento, il limite temporale
# deve comunque sbloccare la situazione.
seq_no_rec = recede(truncated=True, use_receding=False)
check("troncato senza allontanamento rilevato: il timeout sblocca",
      seq_no_rec[-1][1] != "red", f"livello finale {seq_no_rec[-1][1]}")

seq_clean = recede(truncated=False)
check("non troncato: nessuna regressione",
      seq_clean[-1][1] == "green", f"livello finale {seq_clean[-1][1]}")

exit_trunc = next(d for d, lv in seq if lv != "red")
exit_clean = next(d for d, lv in seq_clean if lv != "red")
print(f"  uscita dal rosso: troncato {exit_trunc:.1f} m, "
      f"pulito {exit_clean:.1f} m")

# Il blocco deve pero' continuare a funzionare quando serve davvero:
# un pedone fermo e vicino, con riquadro troncato, resta ROSSO.
am_s = AlertManager()
ts_s = TrackStateStore()
est_s = DistanceEstimator.for_kitti()
level_static = None
for i in range(60):
    h = KITTI_FOCAL_LENGTH_PX * 1.70 / 6.0
    st = ts_s.update(1, (0, 0, 40, int(h)), est_s.estimate(h, 1242), i / 5.0)
    level_static = am_s.evaluate(1, st["smoothed_distance_m"], 0.85,
                                 st["ttc_s"], True, False)
check("pedone fermo e vicino con riquadro troncato resta rosso",
      level_static == "red", f"livello {level_static}")


print("\n\u2550\u2550\u2550 10. Criterio di corridoio \u2550\u2550\u2550")

est = DistanceEstimator(focal_length_px=KITTI_FOCAL_LENGTH_PX)
LARGH = 1242


def box_a(distanza_m, laterale_m):
    """Riquadro di un pedone a una data distanza e scostamento laterale."""
    h = KITTI_FOCAL_LENGTH_PX * 1.70 / distanza_m
    cx = LARGH / 2.0 + laterale_m * h / 1.70
    return (cx - 20, 0.0, cx + 20, h), h


# La stima laterale non deve dipendere dal campo visivo assunto.
for lat in (-4.0, -1.0, 0.0, 2.5, 6.0):
    box, h = box_a(12.0, lat)
    got = est.lateral_offset((box[0] + box[2]) / 2.0, h, LARGH)
    check(f"scostamento laterale {lat:+.1f} m", abs(got - lat) < 0.01,
          f"ottenuto {got:.2f}")

largo = DistanceEstimator(hfov_deg=120.0)
box, h = box_a(12.0, 3.0)
got = largo.lateral_offset((box[0] + box[2]) / 2.0, h, LARGH)
check("la stima laterale non dipende dal campo visivo", abs(got - 3.0) < 0.01,
      f"ottenuto {got:.2f}")


def percorso(laterale_iniziale, velocita_laterale, distanza_m=8.0,
             fotogrammi=40, fps=10.0):
    """Livello finale di un pedone a distanza fissa che si sposta di lato."""
    am = AlertManager()
    ts = TrackStateStore()
    livello = None
    for i in range(fotogrammi):
        t = i / fps
        lat = laterale_iniziale + velocita_laterale * t
        box, h = box_a(distanza_m, lat)
        st = ts.update(1, box, est.estimate(h, LARGH), t,
                       est.lateral_offset((box[0] + box[2]) / 2.0, h, LARGH))
        livello = am.evaluate(1, st["smoothed_distance_m"], 0.85,
                              st["ttc_s"], False, False,
                              st["smoothed_lateral_m"],
                              st["lateral_speed_mps"])
    return livello


check("pedone a 8 m sulla traiettoria: rosso",
      percorso(0.0, 0.0) == "red", f"livello {percorso(0.0, 0.0)}")
check("pedone a 8 m fermo sul marciapiede a 5 m di lato: declassato",
      percorso(5.0, 0.0) == "yellow", f"livello {percorso(5.0, 0.0)}")
check("pedone a 8 m fermo a 1 m di lato: resta rosso",
      percorso(1.0, 0.0) == "red", f"livello {percorso(1.0, 0.0)}")
check("pedone che attraversa da destra verso il centro: rosso",
      percorso(5.0, -1.5) == "red", f"livello {percorso(5.0, -1.5)}")
check("pedone che si allontana lateralmente: declassato",
      percorso(3.0, 1.5) == "yellow", f"livello {percorso(3.0, 1.5)}")
check("senza stima laterale il livello resta prudenziale",
      percorso(0.0, 0.0) == "red")


print(f"\n{'='*50}\n  TOTALE: {ok} superati, {fail} falliti")

sys.exit(1 if fail else 0)
