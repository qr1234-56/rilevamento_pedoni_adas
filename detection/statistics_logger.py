"""
Registrazione su file CSV delle statistiche prodotte durante l'analisi.

Ogni riga riporta, per il frame corrispondente, la distanza in metri del
pedone più vicino e il tempo residuo alla collisione più basso osservato.

Il file registra inoltre, nella prima riga di commento, la configurazione
utilizzata per l'esecuzione: senza questa informazione i CSV prodotti da
versioni diverse del sistema non sono confrontabili fra loro.
"""

import csv
import json
from datetime import datetime
from pathlib import Path


class StatisticsLogger:
    """
    Scrive le statistiche di ciascun frame in un file CSV.
    """

    FIELDNAMES = [
        "timestamp",
        "frame",
        "pedestrians",
        "alert_level",
        "fps",
        "average_confidence",
        "minimum_distance_m",
        "minimum_ttc_s",
    ]

    def __init__(self, output_path, configuration=None):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        # Configurazione in prima riga di commento: i CSV restano confrontabili.
        if configuration:
            self.file.write(
                "# "
                + json.dumps(configuration, ensure_ascii=False, default=str)
                + "\n"
            )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
        )

        self.writer.writeheader()

    @staticmethod
    def _round(value, digits):
        """
        Arrotonda il valore lasciando vuota la cella quando è assente.
        """

        if value is None:
            return ""

        return round(value, digits)

    def log(
        self,
        frame,
        pedestrians,
        alert_level,
        fps,
        average_confidence,
        minimum_distance_m,
        minimum_ttc_s,
    ):
        """
        Registra le statistiche relative a un singolo frame.
        """

        row = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame,
            "pedestrians": pedestrians,
            "alert_level": alert_level,
            "fps": self._round(fps, 2),
            "average_confidence": self._round(average_confidence, 3),
            "minimum_distance_m": self._round(minimum_distance_m, 2),
            "minimum_ttc_s": self._round(minimum_ttc_s, 2),
        }

        self.writer.writerow(row)

        # Scrittura immediata: i dati restano se l'elaborazione si interrompe.
        self.file.flush()

    def close(self):
        """
        Chiude il file CSV.
        """

        if not self.file.closed:
            self.file.close()
