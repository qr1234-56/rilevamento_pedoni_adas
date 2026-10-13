from pathlib import Path
from ultralytics import YOLO


def load_yolo_model(model_path=None):
    """
    Carica il modello YOLO utilizzato dal sistema di rilevamento.

    Se non viene specificato un percorso, viene utilizzato
    automaticamente il modello yolov8n.pt presente nella cartella
    principale del progetto.
    """

    # Cartella del progetto, a partire da questo file.
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent

        model_path = project_root / "yolov8n.pt"

    return YOLO(str(model_path))
