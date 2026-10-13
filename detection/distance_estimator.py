"""
Stima metrica della distanza dei pedoni.

La distanza è espressa in metri e ricavata dal modello pinhole della
telecamera, che lega l'altezza apparente di un oggetto alla sua distanza
secondo la relazione:

    h_px / f = H / d        da cui        d = f * H / h_px

dove:
    h_px = altezza del bounding box in pixel;
    f    = lunghezza focale della telecamera, in pixel;
    H    = altezza reale dell'oggetto, in metri;
    d    = distanza dell'oggetto dalla telecamera, in metri.

Il risultato è una grandezza fisica confrontabile fra video diversi, e
consente di esprimere in metri le soglie di allerta.
"""

import math


# Altezza media assunta per un pedone adulto, in metri.
DEFAULT_PEDESTRIAN_HEIGHT_M = 1.70

# fx della matrice di proiezione P2 di KITTI (immagini 1242x375).
KITTI_FOCAL_LENGTH_PX = 721.5377

# Campo visivo assunto per sorgenti non calibrate.
DEFAULT_HFOV_DEG = 60.0

# Fuori da questo intervallo la stima viene saturata.
MIN_DISTANCE_M = 0.5
MAX_DISTANCE_M = 120.0


def focal_length_from_fov(frame_width_px, hfov_deg=DEFAULT_HFOV_DEG):
    """
    Ricava la lunghezza focale in pixel a partire dal campo visivo
    orizzontale dichiarato.

    Si utilizza quando la telecamera non è calibrata: è il caso della
    webcam e dei video caricati dall'utente.
    """

    # Un campo visivo non valido renderebbe la formula indeterminata.
    if frame_width_px <= 0 or not (0.0 < hfov_deg < 180.0):
        return None

    half_fov_rad = math.radians(hfov_deg) / 2.0

    return (frame_width_px / 2.0) / math.tan(half_fov_rad)


def is_truncated(y1, y2, frame_height, margin_px=2):
    """
    Indica se il bounding box tocca il bordo superiore o inferiore
    dell'immagine.

    Un box troncato sottostima l'altezza reale del pedone, e di conseguenza
    la formula pinhole ne sovrastima la distanza. Poiché un pedone tagliato
    dal bordo inferiore è quasi sempre molto vicino al veicolo, la
    segnalazione viene usata a valle per evitare declassamenti imprudenti
    del livello di allerta.
    """

    return y1 <= margin_px or y2 >= (frame_height - margin_px)


class DistanceEstimator:
    """
    Converte l'altezza in pixel di un bounding box in una distanza in metri.
    """

    def __init__(
        self,
        focal_length_px=None,
        hfov_deg=DEFAULT_HFOV_DEG,
        pedestrian_height_m=DEFAULT_PEDESTRIAN_HEIGHT_M,
    ):
        # Focale nota; se None si ricava dal campo visivo.
        self.focal_length_px = focal_length_px

        self.hfov_deg = hfov_deg

        self.pedestrian_height_m = pedestrian_height_m

    @classmethod
    def for_kitti(cls, pedestrian_height_m=DEFAULT_PEDESTRIAN_HEIGHT_M):
        """
        Crea un estimatore configurato per le immagini KITTI rettificate.
        """

        return cls(
            focal_length_px=KITTI_FOCAL_LENGTH_PX,
            pedestrian_height_m=pedestrian_height_m,
        )

    def focal_length(self, frame_width_px):
        """
        Restituisce la lunghezza focale da utilizzare per il frame corrente.
        """

        if self.focal_length_px is not None:
            return self.focal_length_px

        return focal_length_from_fov(frame_width_px, self.hfov_deg)

    def lateral_offset(self, box_center_x_px, box_height_px, frame_width_px):
        """
        Scostamento laterale del pedone rispetto all'asse ottico, in metri.
        Positivo verso destra.

        Deriva dalla stessa geometria della distanza applicata all'asse
        orizzontale. Nel rapporto la focale si semplifica: il risultato non
        dipende dal campo visivo assunto, ma solo dall'altezza ipotizzata
        del pedone. Si assume inoltre che il punto principale coincida con
        il centro dell'immagine.
        """

        if box_height_px <= 0 or frame_width_px <= 0:
            return None

        return (
            (box_center_x_px - frame_width_px / 2.0)
            * self.pedestrian_height_m
            / float(box_height_px)
        )

    def estimate(self, box_height_px, frame_width_px):
        """
        Stima la distanza di un pedone a partire dall'altezza del suo
        bounding box.

        Restituisce la distanza in metri, oppure None quando i dati
        disponibili non consentono un calcolo sensato.
        """

        # Un box di altezza nulla o negativa non è utilizzabile.
        if box_height_px <= 0:
            return None

        focal_length_px = self.focal_length(frame_width_px)

        # Senza una focale valida la stima non è possibile.
        if not focal_length_px or focal_length_px <= 0:
            return None

        distance_m = (
            focal_length_px * self.pedestrian_height_m
        ) / float(box_height_px)

        return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))
