"""
Stato temporale delle tracce.

L'associazione fra rilevazioni e tracce è affidata a ByteTrack, incluso in
Ultralytics: questo modulo non esegue quindi alcun accoppiamento, ma si
limita a mantenere, per ciascun identificativo assegnato dal tracker, la
storia necessaria a stimare il movimento del pedone.

Le grandezze derivate sono espresse in metri e metri al secondo, e non in
pixel: non dipendono quindi dalla risoluzione del video né dalla frequenza
di elaborazione.
"""


# Media mobile sulla distanza: più basso, più filtra.
DISTANCE_SMOOTHING = 0.35

# Sotto questa velocità il movimento in profondità è trascurabile.
CLOSING_SPEED_DEADBAND_MPS = 0.3

# Media mobile sulla velocità di avvicinamento. La velocità è la derivata
# di una distanza già rumorosa, quindi va filtrata a sua volta: senza,
# un singolo fotogramma anomalo produce un tempo al contatto assurdo.
SPEED_SMOOTHING = 0.30

# Sotto questa velocità lo spostamento laterale è trascurabile.
LATERAL_SPEED_DEADBAND_MPS = 0.2

# Oltre questa variazione d'altezza si sospetta un cambio di postura.
POSTURE_SIZE_CHANGE = 0.20

# Tolleranza sul bordo inferiore, in frazione dell'altezza.
POSTURE_BOTTOM_TOLERANCE = 0.15


class TrackStateStore:
    """
    Mantiene lo stato temporale delle tracce prodotte dal tracker.
    """

    def __init__(
        self,
        distance_smoothing=DISTANCE_SMOOTHING,
        closing_speed_deadband=CLOSING_SPEED_DEADBAND_MPS,
        speed_smoothing=SPEED_SMOOTHING,
    ):
        self.distance_smoothing = distance_smoothing

        self.speed_smoothing = speed_smoothing
        self.closing_speed_deadband = closing_speed_deadband

        self._states = {}

    def reset(self):
        """
        Azzera lo stato di tutte le tracce.
        """

        self._states = {}

    def prune(self, active_track_ids):
        """
        Elimina lo stato delle tracce non più presenti nella scena.
        """

        active = set(active_track_ids)

        for track_id in list(self._states):
            if track_id not in active:
                del self._states[track_id]

    def update(self, track_id, box, distance_m, timestamp, lateral_m=None):
        """
        Aggiorna lo stato di una traccia e restituisce le grandezze derivate.

        Parametri:
            track_id    identificativo assegnato dal tracker;
            box         tupla (x1, y1, x2, y2) in pixel;
            distance_m  distanza stimata in metri, oppure None;
            timestamp   istante dell'osservazione, in secondi;
            lateral_m   scostamento laterale in metri, oppure None.

        Restituisce un dizionario con:
            smoothed_distance_m  distanza filtrata;
            closing_speed_mps    velocità di avvicinamento (positiva se
                                 il pedone si avvicina);
            ttc_s                tempo residuo alla collisione, o None;
            smoothed_lateral_m   scostamento laterale filtrato;
            lateral_speed_mps    velocità laterale, positiva verso destra;
            depth_direction      "approaching" | "receding" | "stable";
            size_change_ratio    variazione relativa dell'altezza del box;
            posture_change       True se la variazione sembra dovuta a un
                                 cambio di postura;
            movement_x/y         spostamento del centro, in pixel;
            movement_distance    modulo dello spostamento, in pixel.
        """

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        height = max(1.0, float(y2 - y1))
        bottom_y = float(y2)

        previous = self._states.get(track_id)


        if previous is None:
            self._states[track_id] = {
                "center": (center_x, center_y),
                "height": height,
                "bottom_y": bottom_y,
                "distance": distance_m,
                "lateral": lateral_m,
                "speed": 0.0,
                "timestamp": timestamp,
            }

            return {
                "smoothed_distance_m": distance_m,
                "closing_speed_mps": 0.0,
                "ttc_s": None,
                "smoothed_lateral_m": lateral_m,
                "lateral_speed_mps": 0.0,
                "depth_direction": "stable",
                "size_change_ratio": 0.0,
                "posture_change": False,
                "movement_x": 0.0,
                "movement_y": 0.0,
                "movement_distance": 0.0,
            }


        previous_center = previous["center"]

        movement_x = center_x - previous_center[0]
        movement_y = center_y - previous_center[1]
        movement_distance = (movement_x ** 2 + movement_y ** 2) ** 0.5


        previous_height = previous["height"]
        size_change_ratio = (height - previous_height) / previous_height

        bottom_change = bottom_y - previous["bottom_y"]

        # Altezza che cambia con bordo inferiore fermo: cambio di postura.
        posture_change = (
            abs(size_change_ratio) >= POSTURE_SIZE_CHANGE
            and abs(bottom_change) < POSTURE_BOTTOM_TOLERANCE * height
        )


        previous_distance = previous["distance"]

        if distance_m is None:
            smoothed_distance = previous_distance

        elif previous_distance is None:
            smoothed_distance = distance_m

        else:
            alpha = self.distance_smoothing
            smoothed_distance = (
                alpha * distance_m + (1.0 - alpha) * previous_distance
            )


        previous_lateral = previous["lateral"]

        if lateral_m is None:
            smoothed_lateral = previous_lateral

        elif previous_lateral is None:
            smoothed_lateral = lateral_m

        else:
            alpha = self.distance_smoothing
            smoothed_lateral = (
                alpha * lateral_m + (1.0 - alpha) * previous_lateral
            )


        elapsed = timestamp - previous["timestamp"]

        lateral_speed = 0.0

        closing_speed = 0.0
        ttc = None

        if (
            elapsed > 0
            and smoothed_distance is not None
            and previous_distance is not None
        ):
            # Positiva quando la distanza diminuisce, cioè in avvicinamento.
            istantanea = (previous_distance - smoothed_distance) / elapsed

            # Dopo un cambio di postura la velocità non è attendibile.
            if posture_change:
                istantanea = 0.0

            beta = self.speed_smoothing
            closing_speed = (
                beta * istantanea + (1.0 - beta) * previous["speed"]
            )

            # Il tempo residuo ha senso solo in avvicinamento effettivo.
            if closing_speed > self.closing_speed_deadband:
                ttc = smoothed_distance / closing_speed

        if (
            elapsed > 0
            and smoothed_lateral is not None
            and previous_lateral is not None
            and not posture_change
        ):
            lateral_speed = (smoothed_lateral - previous_lateral) / elapsed

            if abs(lateral_speed) <= LATERAL_SPEED_DEADBAND_MPS:
                lateral_speed = 0.0


        if posture_change or abs(closing_speed) <= self.closing_speed_deadband:
            depth_direction = "stable"

        elif closing_speed > 0:
            depth_direction = "approaching"

        else:
            depth_direction = "receding"


        self._states[track_id] = {
            "center": (center_x, center_y),
            "height": height,
            "bottom_y": bottom_y,
            "distance": smoothed_distance,
            "lateral": smoothed_lateral,
            "speed": closing_speed,
            "timestamp": timestamp,
        }

        return {
            "smoothed_distance_m": smoothed_distance,
            "closing_speed_mps": closing_speed,
            "ttc_s": ttc,
            "smoothed_lateral_m": smoothed_lateral,
            "lateral_speed_mps": lateral_speed,
            "depth_direction": depth_direction,
            "size_change_ratio": size_change_ratio,
            "posture_change": posture_change,
            "movement_x": movement_x,
            "movement_y": movement_y,
            "movement_distance": movement_distance,
        }
