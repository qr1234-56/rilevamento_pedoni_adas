from pathlib import Path
import subprocess
import time
import wave

import cv2

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from detection.detector import PedestrianDetector
from detection.statistics_logger import StatisticsLogger


router = APIRouter()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "detected_"

# Intervallo, in secondi, fra due segnalazioni acustiche mentre il livello
# resta ROSSO.
RED_REPEAT_S = 0.6

MODEL_PATH = "yolov8n.pt"

detector = PedestrianDetector(
    confidence_threshold=0.25,
    model_path=MODEL_PATH,
)

templates = Jinja2Templates(directory="frontend/templates")


@router.get("/")
def home(request: Request):
    """
    Restituisce la pagina principale della dashboard.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


def generate_frames(source=0):
    """
    Acquisisce i frame dalla webcam e li invia al browser
    come stream MJPEG.
    """

    cap = cv2.VideoCapture(source)

    # Risoluzione contenuta per l'elaborazione in tempo reale.
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Azzera lo stato interno: le tracce non proseguono da una sessione all'altra.
    detector.reset()

    # stop() disattiva l'audio: va riabilitato a ogni nuova sessione.
    detector.audio_alert.start()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            result = detector.process(frame)
            processed_frame = result["frame"]

            success, buffer = cv2.imencode(".jpg", processed_frame)

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        cap.release()

        detector.audio_alert.stop()


@router.get("/video_feed")
def video_feed():
    """
    Endpoint utilizzato dalla dashboard per lo streaming della webcam.
    """

    return StreamingResponse(
        generate_frames(0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/stop-webcam")
def stop_webcam():
    """
    Arresta l'allarme audio quando la webcam viene disattivata.
    """

    detector.audio_alert.stop()

    return {
        "status": "ok",
        "message": "Webcam arrestata",
    }


def create_alert_audio_track(
    output_path,
    duration,
    alert_times,
    sound_path,
    sample_rate=44100,
):
    """
    Costruisce una traccia WAV lunga quanto il video, con il suono
    dell'allarme inserito nei momenti indicati da alert_times.
    """

    with wave.open(str(sound_path), "rb") as sound_file:
        channels = sound_file.getnchannels()
        width = sound_file.getsampwidth()
        source_rate = sound_file.getframerate()
        sound_frames = sound_file.readframes(sound_file.getnframes())

    # Formato diverso dal previsto: converte FFmpeg più avanti.
    if channels not in (1, 2):
        channels = 1

    # Riporta l'allarme alla frequenza della traccia.
    if source_rate != sample_rate:
        temp_sound = OUTPUT_DIR / "_temp_alert_source.wav"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(sound_path),
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(temp_sound),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with wave.open(str(temp_sound), "rb") as converted_file:
            width = converted_file.getsampwidth()
            sound_frames = converted_file.readframes(
                converted_file.getnframes()
            )

        temp_sound.unlink(missing_ok=True)

    # Frame audio necessari a coprire il video.
    total_frames = max(
        1,
        int(duration * sample_rate),
    )

    silence = bytearray(total_frames * channels * width)

    sound_duration = len(sound_frames) / (
        source_rate * channels * width
    )

    # Inserisce l'allarme in corrispondenza degli eventi RED.
    for alert_time in alert_times:

        start_frame = int(alert_time * sample_rate)

        if start_frame >= total_frames:
            continue

        alert_frame_count = min(
            int(sound_duration * sample_rate),
            total_frames - start_frame,
        )

        bytes_per_frame = channels * width

        start_byte = start_frame * bytes_per_frame
        end_byte = (
            start_byte
            + alert_frame_count * bytes_per_frame
        )

        alert_bytes = sound_frames[
            :alert_frame_count * bytes_per_frame
        ]

        silence[start_byte:end_byte] = alert_bytes

    with wave.open(str(output_path), "wb") as output_file:
        output_file.setnchannels(channels)
        output_file.setsampwidth(width)
        output_file.setframerate(sample_rate)
        output_file.writeframes(silence)


def add_audio_to_video(video_path, audio_path, final_path):
    """
    Utilizza FFmpeg per inserire la traccia audio nel video elaborato.
    """

    # FFmpeg ricodifica solo l'audio, il video resta invariato.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@router.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """
    Riceve un video, lo analizza frame per frame e restituisce il video
    elaborato, con l'audio dell'allarme inserito in corrispondenza dei
    momenti in cui il livello è salito a ROSSO.
    """

    if not file.filename:
        return {
            "status": "error",
            "message": "Nome file non valido.",
        }

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        return {
            "status": "error",
            "message": "Formato video non supportato.",
        }

    # Video già analizzato: rianalizzarlo sovrappone due serie di riquadri.
    if Path(file.filename).name.startswith(OUTPUT_PREFIX):
        original_name = Path(file.filename).name[len(OUTPUT_PREFIX):]

        return {
            "status": "error",
            "message": (
                "Questo file e' gia' il risultato di un'analisi precedente e "
                "contiene i riquadri disegnati sopra le immagini. "
                f"Carica il video originale ({original_name})."
            ),
        }

    content = await file.read()

    if not content:
        return {
            "status": "error",
            "message": "Il file caricato è vuoto.",
        }

    input_path = UPLOAD_DIR / Path(file.filename).name
    input_path.write_bytes(content)

    temporary_output_path = (
        OUTPUT_DIR / f"_temp_{input_path.name}"
    )

    output_filename = f"{OUTPUT_PREFIX}{input_path.name}"
    output_path = OUTPUT_DIR / output_filename

    selected_detector = detector

    # Azzera lo stato interno prima di una nuova analisi.
    selected_detector.reset()

    # L'analisi offline non suona: l'allarme finisce nella traccia audio.
    selected_detector.audio_alert.stop()

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        return {
            "status": "error",
            "message": "Impossibile aprire il video.",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temporary_output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        return {
            "status": "error",
            "message": "Impossibile creare il video di output.",
        }

    statistics_path = (
        OUTPUT_DIR / f"{input_path.stem}_statistics.csv"
    )

    # La configurazione rende i CSV confrontabili nel tempo.
    run_configuration = {
        "model": MODEL_PATH,
        "tracker": selected_detector.tracker,
        "confidence_threshold": selected_detector.confidence_threshold,
        "focal_length_px": selected_detector.distance_estimator.focal_length(
            width
        ),
        "pedestrian_height_m": (
            selected_detector.distance_estimator.pedestrian_height_m
        ),
        "red_distance_m": selected_detector.alert_manager.red_distance_m,
        "yellow_distance_m": selected_detector.alert_manager.yellow_distance_m,
        "red_ttc_s": selected_detector.alert_manager.red_ttc_s,
        "corridor_half_width_m": (
            selected_detector.alert_manager.corridor_half_width_m
        ),
        "source": input_path.name,
        "video_fps": fps,
        "frame_size": [width, height],
    }

    logger = StatisticsLogger(statistics_path, configuration=run_configuration)

    frame_count = 0
    total_pedestrians = 0

    green_frames = 0
    yellow_frames = 0
    red_frames = 0

    red_alert_times = []
    last_red_alert_time = -RED_REPEAT_S

    previous_alert_level = "green"

    processing_start = time.time()

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            # Tempo del video, non dell'orologio: l'elaborazione non procede
            # alla velocità di riproduzione.
            result = selected_detector.process(
                frame,
                timestamp=frame_count / fps,
            )

            processed_frame = result["frame"]
            pedestrians_count = result["pedestrians"]
            processed_fps = result["fps"]
            alert_level = result["alert_level"]
            average_confidence = result["average_confidence"]
            minimum_distance_m = result["minimum_distance_m"]
            minimum_ttc_s = result["minimum_ttc_s"]

            writer.write(processed_frame)

            frame_count += 1

            total_pedestrians += pedestrians_count

            if alert_level == "green":
                green_frames += 1

            elif alert_level == "yellow":
                yellow_frames += 1

            elif alert_level == "red":
                red_frames += 1

                current_time = (frame_count - 1) / fps

                # Allarme all'ingresso in ROSSO e poi a intervalli regolari
                # finché la condizione persiste.
                if (
                    previous_alert_level != "red"
                    or current_time - last_red_alert_time >= RED_REPEAT_S
                ):
                    red_alert_times.append(current_time)
                    last_red_alert_time = current_time

            previous_alert_level = alert_level

            logger.log(
                frame=frame_count,
                pedestrians=pedestrians_count,
                alert_level=alert_level,
                fps=processed_fps,
                average_confidence=average_confidence,
                minimum_distance_m=minimum_distance_m,
                minimum_ttc_s=minimum_ttc_s,
            )

    finally:
        cap.release()
        writer.release()

        logger.close()

        # Sicurezza: garantisce che nessun audio rimanga attivo.
        selected_detector.audio_alert.stop()

    processing_end = time.time()

    processing_time = processing_end - processing_start

    if processing_time > 0:
        average_fps = frame_count / processing_time
    else:
        average_fps = 0.0

    if frame_count > 0:
        average_pedestrians = total_pedestrians / frame_count
    else:
        average_pedestrians = 0.0


    sound_path = (
        Path(__file__).resolve().parent.parent
        / "sounds"
        / "red_alert.wav"
    )

    video_duration = frame_count / fps if fps > 0 else 0.0

    alert_audio_path = (
        OUTPUT_DIR / f"_temp_alert_{input_path.stem}.wav"
    )

    if red_alert_times:
        create_alert_audio_track(
            output_path=alert_audio_path,
            duration=video_duration,
            alert_times=red_alert_times,
            sound_path=sound_path,
        )

        add_audio_to_video(
            video_path=temporary_output_path,
            audio_path=alert_audio_path,
            final_path=output_path,
        )

        alert_audio_path.unlink(missing_ok=True)

        temporary_output_path.unlink(missing_ok=True)

    else:
        # Nessun RED: il video temporaneo diventa quello finale.
        temporary_output_path.replace(output_path)

    return {
        "status": "success",
        "input_file": input_path.name,
        "output_file": output_path.name,
        "frames_processed": frame_count,
        "pedestrians_detected": total_pedestrians,
        "green_frames": green_frames,
        "yellow_frames": yellow_frames,
        "red_frames": red_frames,
        "average_pedestrians": round(
            average_pedestrians,
            2,
        ),
        "average_fps": round(
            average_fps,
            2,
        ),
        "processing_time_seconds": round(
            processing_time,
            2,
        ),
        "statistics_file": statistics_path.name,
    }


@router.get("/download-video/{filename}")
def download_video(filename: str):
    """
    Permette di scaricare il video elaborato.
    """

    # Impedisce percorsi fuori dalla cartella outputs.
    file_path = OUTPUT_DIR / Path(filename).name

    if not file_path.exists():
        return {
            "status": "error",
            "message": "File non trovato.",
        }

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
    )
