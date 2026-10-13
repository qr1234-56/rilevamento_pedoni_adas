"""
Modulo principale per il rilevamento dei pedoni.

Coordina il modello YOLO, il tracker ByteTrack, la stima della distanza,
il sistema di allerta a tre livelli e la visualizzazione sul frame.

L'associazione fra rilevazioni e tracce è affidata a ByteTrack, integrato
in Ultralytics: la sua seconda passata di associazione recupera le
rilevazioni a bassa confidenza, tipiche dei pedoni parzialmente occlusi.
"""

import time

import cv2

from detection.alert_manager import AlertManager
from detection.audio_alert import AudioAlert
from detection.distance_estimator import DistanceEstimator, is_truncated
from detection.model_loader import load_yolo_model
from detection.track_state import TrackStateStore


# Più bassa della soglia operativa: ByteTrack usa le rilevazioni deboli.
TRACKER_CONFIDENCE = 0.15

# Valore predefinito di Ultralytics: abbassarlo fonde pedoni vicini fra loro.
NMS_IOU = 0.7

# Colori associati ai livelli di allerta, in formato BGR.
LEVEL_COLORS = {
    AlertManager.GREEN: (0, 255, 0),
    AlertManager.YELLOW: (0, 255, 255),
    AlertManager.RED: (0, 0, 255),
}

# Nomi dei livelli mostrati sul frame. I caratteri di OpenCV non
# supportano le lettere accentate: i testi restano senza accenti.
LEVEL_LABELS = {
    AlertManager.GREEN: "VERDE",
    AlertManager.YELLOW: "GIALLO",
    AlertManager.RED: "ROSSO",
}


class PedestrianDetector:
    """
    Rilevatore di pedoni con stima della distanza e allerta a tre livelli.
    """

    def __init__(
        self,
        confidence_threshold=0.4,
        model_path=None,
        distance_estimator=None,
        alert_manager=None,
        tracker="bytetrack.yaml",
    ):
        # Soglia operativa: sotto questo valore la traccia non viene mostrata.
        self.confidence_threshold = confidence_threshold

        self.model = load_yolo_model(model_path)

        self.tracker = tracker

        self.distance_estimator = (
            distance_estimator
            if distance_estimator is not None
            else DistanceEstimator()
        )

        self.alert_manager = (
            alert_manager
            if alert_manager is not None
            else AlertManager()
        )

        self.audio_alert = AudioAlert()

        self.track_state = TrackStateStore()

        self.previous_time = None
        self.smoothed_fps = 0.0

        # Fattore della media mobile applicata all'FPS visualizzato.
        self.fps_alpha = 0.10


    def reset(self):
        """
        Reimposta lo stato interno prima di una nuova analisi.
        """

        self.previous_time = None
        self.smoothed_fps = 0.0

        self.track_state.reset()
        self.alert_manager.reset()

        # Azzera il tracker: gli identificativi non proseguono da un video all'altro.
        self._reset_ultralytics_tracker()

    def _reset_ultralytics_tracker(self):
        """
        Azzera il tracker di Ultralytics.

        L'API cambia fra le versioni della libreria, per cui si tenta prima
        il metodo dedicato e, in mancanza, si forza la ricostruzione del
        predictor alla chiamata successiva.
        """

        predictor = getattr(self.model, "predictor", None)

        if predictor is None:
            return

        trackers = getattr(predictor, "trackers", None)

        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)

                if callable(reset):
                    reset()

            return

        # Fallback: alla prossima chiamata il predictor viene ricreato.
        self.model.predictor = None


    def _update_fps(self, current_time):
        """
        Aggiorna e restituisce l'FPS di elaborazione, filtrato.
        """

        if self.previous_time is None:
            self.previous_time = current_time

        elapsed = current_time - self.previous_time
        self.previous_time = current_time

        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        if self.smoothed_fps == 0.0:
            self.smoothed_fps = instant_fps
        else:
            self.smoothed_fps = (
                self.fps_alpha * instant_fps
                + (1.0 - self.fps_alpha) * self.smoothed_fps
            )

        return self.smoothed_fps


    def process(self, frame, timestamp=None):
        """
        Analizza un singolo frame.

        Restituisce un dizionario contenente il frame annotato, il numero di
        pedoni, l'FPS di elaborazione, il livello di allerta complessivo, la
        confidenza media, la distanza minima stimata e il tempo residuo alla
        collisione più basso osservato nel frame.

        timestamp è l'istante del frame in secondi. Per un video caricato
        va passato il tempo del video (indice del frame / fps): l'orologio
        del computer misurerebbe il tempo di elaborazione, non quello
        trascorso nella scena, e falserebbe velocità e tempo al contatto.
        Se è None (webcam) si usa l'orologio di sistema.
        """

        # L'FPS misura la velocità di elaborazione: usa sempre l'orologio.
        wall_time = time.time()
        fps = self._update_fps(wall_time)

        current_time = wall_time if timestamp is None else timestamp

        frame_height, frame_width = frame.shape[:2]


        # classes=[0]: solo persone. persist=True: tracce mantenute fra i frame.
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=TRACKER_CONFIDENCE,
            iou=NMS_IOU,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = getattr(result, "boxes", None)

            if boxes is None or boxes.id is None:
                continue

            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            track_ids = boxes.id.int().cpu().numpy()

            for box, confidence, track_id in zip(
                coordinates,
                confidences,
                track_ids,
            ):
                confidence = float(confidence)

                # Sotto soglia: la traccia resta nel tracker ma non nel risultato.
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = (int(value) for value in box)

                detections.append(
                    {
                        "track_id": int(track_id),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                    }
                )


        active_ids = [detection["track_id"] for detection in detections]

        # Libera lo stato delle tracce non più presenti.
        self.track_state.prune(active_ids)
        self.alert_manager.prune(active_ids)

        confidences = []
        distances = []
        times_to_collision = []
        levels = []

        for detection in detections:
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            box_height = y2 - y1

            distance_m = self.distance_estimator.estimate(
                box_height,
                frame_width,
            )

            # Scostamento dall'asse della telecamera, per il criterio di corridoio.
            lateral_m = self.distance_estimator.lateral_offset(
                (x1 + x2) / 2.0,
                box_height,
                frame_width,
            )

            # Riquadro sul bordo: altezza sottostimata, distanza sovrastimata.
            truncated = is_truncated(y1, y2, frame_height)

            state = self.track_state.update(
                track_id=detection["track_id"],
                box=(x1, y1, x2, y2),
                distance_m=distance_m,
                timestamp=current_time,
                lateral_m=lateral_m,
            )

            smoothed_distance = state["smoothed_distance_m"]
            ttc = state["ttc_s"]

            level = self.alert_manager.evaluate(
                track_id=detection["track_id"],
                distance_m=smoothed_distance,
                confidence=detection["confidence"],
                ttc_s=ttc,
                truncated=truncated,
                receding=(state["depth_direction"] == "receding"),
                lateral_m=state["smoothed_lateral_m"],
                lateral_speed_mps=state["lateral_speed_mps"],
            )

            detection.update(state)
            detection["distance_m"] = smoothed_distance
            detection["truncated"] = truncated
            detection["alert_level"] = level

            confidences.append(detection["confidence"])
            levels.append(level)

            if smoothed_distance is not None:
                distances.append(smoothed_distance)

            if ttc is not None:
                times_to_collision.append(ttc)

        # Livello complessivo del frame: il più grave fra quelli dei pedoni.
        overall_alert = AlertManager.most_severe(levels)


        for detection in detections:
            self._draw_detection(frame, detection)


        if overall_alert == AlertManager.RED:
            self.audio_alert.play()


        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Distanza del pedone più vicino e tempo residuo più critico.
        minimum_distance = min(distances) if distances else None
        minimum_ttc = min(times_to_collision) if times_to_collision else None

        self._draw_hud(
            frame,
            fps=fps,
            pedestrians=len(detections),
            overall_alert=overall_alert,
            average_confidence=average_confidence,
            minimum_distance=minimum_distance,
            minimum_ttc=minimum_ttc,
        )

        # Bordo rosso di sicurezza quando la situazione è critica.
        if overall_alert == AlertManager.RED:
            cv2.rectangle(
                frame,
                (0, 0),
                (frame_width - 1, frame_height - 1),
                LEVEL_COLORS[AlertManager.RED],
                8,
            )

        return {
            "frame": frame,
            "pedestrians": len(detections),
            "fps": fps,
            "alert_level": overall_alert,
            "average_confidence": average_confidence,
            "minimum_distance_m": minimum_distance,
            "minimum_ttc_s": minimum_ttc,
            "detections": detections,
        }


    def _draw_detection(self, frame, detection):
        """
        Disegna il bounding box, l'etichetta e la direzione del movimento.
        """

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        color = LEVEL_COLORS.get(
            detection["alert_level"],
            LEVEL_COLORS[AlertManager.GREEN],
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        distance_m = detection.get("distance_m")

        if distance_m is None:
            distance_text = "n/d"
        else:
            distance_text = f"{distance_m:.1f}m"

        # La distanza di un box troncato è poco attendibile: viene segnalata.
        if detection.get("truncated"):
            distance_text += "?"

        label = (
            f"ID:{detection['track_id']} "
            f"{distance_text} "
            f"{detection['confidence']:.2f} "
            f"{LEVEL_LABELS.get(detection['alert_level'], detection['alert_level'].upper())}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        # Freccia che indica lo spostamento del pedone nel piano immagine.
        if detection.get("movement_distance", 0.0) > 2:
            current_center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
            )

            previous_center = (
                int(current_center[0] - detection["movement_x"]),
                int(current_center[1] - detection["movement_y"]),
            )

            cv2.arrowedLine(
                frame,
                previous_center,
                current_center,
                color,
                2,
                tipLength=0.3,
            )

    def _draw_hud(
        self,
        frame,
        fps,
        pedestrians,
        overall_alert,
        average_confidence,
        minimum_distance,
        minimum_ttc,
    ):
        """
        Disegna il pannello informativo nella parte superiore del frame.
        """

        distance_text = (
            f"{minimum_distance:.1f} m"
            if minimum_distance is not None
            else "n/d"
        )

        ttc_text = (
            f"{minimum_ttc:.1f} s"
            if minimum_ttc is not None
            else "n/d"
        )

        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Pedoni: {pedestrians}",
            f"Allerta: {LEVEL_LABELS.get(overall_alert, overall_alert.upper())}",
            f"Confidenza media: {average_confidence:.2f}",
            f"Distanza minima: {distance_text}",
            f"Tempo al contatto min.: {ttc_text}",
        ]

        # Fascia scura semitrasparente: senza, il testo bianco sparisce
        # sugli sfondi chiari.
        larghezza = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            for line in hud_lines
        )
        altezza = 28 * len(hud_lines) + 16

        pannello = frame[0:altezza, 0:larghezza + 40].copy()
        cv2.rectangle(
            pannello,
            (0, 0),
            (pannello.shape[1], pannello.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(
            pannello, 0.45, frame[0:altezza, 0:larghezza + 40], 0.55, 0,
            frame[0:altezza, 0:larghezza + 40],
        )

        hud_y = 30

        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (20, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            hud_y += 28
