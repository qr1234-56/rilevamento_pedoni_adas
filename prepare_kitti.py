import argparse
import random
import shutil
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_SOURCE = (
    PROJECT_DIR
    / "dataset"
    / "kitti"
    / "raw"
    / "data_object_image_2"
    / "training"
    / "image_2"
)

LABEL_SOURCE = (
    PROJECT_DIR
    / "dataset"
    / "kitti"
    / "raw"
    / "data_object_label_2"
    / "training"
    / "label_2"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "kitti"
)

TRAIN_IMAGE_DIR = OUTPUT_DIR / "images" / "train"
TRAIN_LABEL_DIR = OUTPUT_DIR / "labels" / "train"

VAL_IMAGE_DIR = OUTPUT_DIR / "images" / "val"
VAL_LABEL_DIR = OUTPUT_DIR / "labels" / "val"

DATA_YAML = OUTPUT_DIR / "data.yaml"

# Quota destinata alla validazione.
VAL_RATIO = 0.20

# Seme fisso: lo split è riproducibile.
RANDOM_SEED = 42

# Nome della classe Pedestrian utilizzata nelle annotazioni KITTI.
KITTI_PEDESTRIAN_CLASS = "Pedestrian"

# Indice della classe "person" nel formato YOLO.
YOLO_PERSON_CLASS = 0


def parse_kitti_label(label_path, image_width, image_height):
    """
    Legge un file di annotazione KITTI e converte le bounding box
    della classe Pedestrian nel formato richiesto da YOLO.

    Formato KITTI:
        class truncation occlusion alpha
        xmin ymin xmax ymax
        ...

    Formato YOLO:
        class x_center y_center width height
    """

    yolo_annotations = []

    with label_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) < 8:
                print(
                    f"WARNING: label non valida "
                    f"{label_path.name}, riga {line_number}"
                )
                continue

            object_class = parts[0]

            if object_class != KITTI_PEDESTRIAN_CLASS:
                continue

            try:
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])

            except ValueError:
                print(
                    f"WARNING: coordinate non valide "
                    f"{label_path.name}, riga {line_number}"
                )
                continue

            if xmax <= xmin or ymax <= ymin:
                continue


            # Centro normalizzato rispetto alle dimensioni dell'immagine.
            x_center = ((xmin + xmax) / 2.0) / image_width
            y_center = ((ymin + ymax) / 2.0) / image_height

            # Larghezza e altezza normalizzate fra 0 e 1.
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            yolo_annotations.append(
                f"{YOLO_PERSON_CLASS} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    return yolo_annotations


def get_valid_samples():
    """
    Individua le immagini che possiedono una label corrispondente
    e almeno una annotazione Pedestrian.
    """

    if not IMAGE_SOURCE.exists():
        raise FileNotFoundError(
            f"Cartella immagini non trovata:\n{IMAGE_SOURCE}"
        )

    if not LABEL_SOURCE.exists():
        raise FileNotFoundError(
            f"Cartella label non trovata:\n{LABEL_SOURCE}"
        )

    # Immagini PNG in ordine numerico.
    image_files = sorted(
        IMAGE_SOURCE.glob("*.png")
    )

    valid_samples = []

    missing_labels = 0
    images_without_pedestrians = 0
    invalid_images = 0
    total_pedestrians = 0

    for image_path in image_files:

        label_path = LABEL_SOURCE / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            invalid_images += 1
            continue

        image_height, image_width = image.shape[:2]

        annotations = parse_kitti_label(
            label_path,
            image_width,
            image_height,
        )

        if not annotations:
            images_without_pedestrians += 1
            continue

        valid_samples.append(
            {
                "image": image_path,
                "label": label_path,
                "annotations": annotations,
                "pedestrians": len(annotations),
            }
        )

        total_pedestrians += len(annotations)


    print()
    print("=" * 60)
    print("KITTI DATASET ANALYSIS")
    print("=" * 60)

    print(f"Immagini trovate:              {len(image_files)}")
    print(f"Immagini valide con pedoni:    {len(valid_samples)}")
    print(f"Pedoni totali:                 {total_pedestrians}")
    print(f"Label mancanti:                {missing_labels}")
    print(f"Immagini senza Pedestrian:     {images_without_pedestrians}")
    print(f"Immagini non valide:            {invalid_images}")

    return valid_samples


def write_data_yaml():
    """
    Crea il file di configurazione del dataset utilizzato da Ultralytics.
    """

    content = """path: .

train: images/train

val: images/val

names:

  0: person

"""

    DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )


def prepare_dataset(dry_run=False):
    """
    Crea il dataset YOLO suddividendo i campioni disponibili
    in training set e validation set.
    """

    samples = get_valid_samples()

    if not samples:
        raise RuntimeError(
            "Nessuna immagine valida contenente Pedestrian trovata."
        )


    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    split_index = int(
        len(samples) * (1.0 - VAL_RATIO)
    )

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    train_pedestrians = sum(
        sample["pedestrians"]
        for sample in train_samples
    )

    val_pedestrians = sum(
        sample["pedestrians"]
        for sample in val_samples
    )


    print()
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(
        f"Train: {len(train_samples)} immagini "
        f"({len(train_samples) / len(samples) * 100:.1f}%)"
    )

    print(
        f"Validation: {len(val_samples)} immagini "
        f"({len(val_samples) / len(samples) * 100:.1f}%)"
    )

    print(f"Pedoni train:       {train_pedestrians}")
    print(f"Pedoni validation:  {val_pedestrians}")


    # dry-run: analizza senza copiare nulla.
    if dry_run:
        print()
        print("DRY RUN: nessun file è stato copiato.")
        return


    TRAIN_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAIN_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VAL_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VAL_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    for sample in train_samples:

        destination_image = (
            TRAIN_IMAGE_DIR
            / sample["image"].name
        )

        destination_label = (
            TRAIN_LABEL_DIR
            / f"{sample['image'].stem}.txt"
        )

        shutil.copy2(
            sample["image"],
            destination_image,
        )

        destination_label.write_text(
            "\n".join(sample["annotations"]) + "\n",
            encoding="utf-8",
        )


    for sample in val_samples:

        destination_image = (
            VAL_IMAGE_DIR
            / sample["image"].name
        )

        destination_label = (
            VAL_LABEL_DIR
            / f"{sample['image'].stem}.txt"
        )

        shutil.copy2(
            sample["image"],
            destination_image,
        )

        destination_label.write_text(
            "\n".join(sample["annotations"]) + "\n",
            encoding="utf-8",
        )

    write_data_yaml()


    print()
    print("=" * 60)
    print("DATASET CREATO")
    print("=" * 60)

    print(f"Train images: {TRAIN_IMAGE_DIR}")
    print(f"Train labels: {TRAIN_LABEL_DIR}")
    print(f"Val images:   {VAL_IMAGE_DIR}")
    print(f"Val labels:   {VAL_LABEL_DIR}")
    print(f"YAML:         {DATA_YAML}")


def main():

    parser = argparse.ArgumentParser(
        description="Convert KITTI Pedestrian labels to YOLO format."
    )

    # --dry-run: analizza senza scrivere su disco.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analizza il dataset senza copiare file.",
    )

    args = parser.parse_args()

    prepare_dataset(
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
