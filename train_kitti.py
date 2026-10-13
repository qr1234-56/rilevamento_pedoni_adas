from ultralytics import YOLO


# Modello di partenza.
MODEL_PATH = "yolov8n.pt"

# Configurazione del dataset KITTI convertito.
DATASET_PATH = "dataset/kitti/data.yaml"


def main():


    # Modello pre-addestrato, base per il fine-tuning.
    model = YOLO(MODEL_PATH)


    model.train(

        data=DATASET_PATH,

        epochs=10,

        imgsz=640,

        batch=4,

        device="cpu",

        # 0: caricamento dati nel processo principale.
        workers=0,

        # Cartella dei risultati.
        project="runs",

        name="kitti_test",

        # Riusa la cartella se esiste già.
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
