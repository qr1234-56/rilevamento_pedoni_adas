from pathlib import Path
import platform
import shutil
import subprocess
import threading


class AudioAlert:
    """
    Gestisce la riproduzione dell'allarme acustico associato
    al livello di allerta ROSSO.
    """

    def __init__(self, sound_path=None):
        # Suono predefinito nella cartella sounds.
        if sound_path is None:
            project_root = Path(__file__).resolve().parent.parent
            sound_path = project_root / "sounds" / "red_alert.wav"

        self.sound_path = str(sound_path)

        self._is_playing = False

        # Contiene il processo utilizzato da macOS per riprodurre il suono.
        self._process = None

        self._stop_requested = False

        self._enabled = True

        # Lock: lo stato è letto e scritto da più thread.
        self._lock = threading.Lock()

    def start(self):
        """
        Abilita la riproduzione dell'allarme.

        Viene utilizzato quando inizia una nuova analisi, ad esempio
        quando viene attivata la webcam o avviata l'analisi di un video.
        """

        with self._lock:
            self._enabled = True
            self._stop_requested = False

    def play(self):
        """
        Avvia la riproduzione dell'allarme in modo non bloccante.

        Se un allarme è già in riproduzione oppure l'audio è disabilitato,
        non viene avviato un secondo processo.
        """

        with self._lock:
            if not self._enabled or self._is_playing:
                return

            self._is_playing = True
            self._stop_requested = False

        # Thread separato per non bloccare l'elaborazione.
        thread = threading.Thread(
            target=self._play_sound,
            daemon=True,
        )
        thread.start()

    def stop(self):
        """
        Interrompe immediatamente l'eventuale allarme in riproduzione.

        Questa funzione viene utilizzata, ad esempio, quando l'utente
        disattiva la webcam.
        """

        with self._lock:
            self._enabled = False

            self._stop_requested = True

            process = self._process

            self._is_playing = False

        if process is not None and process.poll() is None:
            process.terminate()

    def _player_command(self):
        """
        Restituisce il comando di riproduzione adatto al sistema operativo,
        oppure None se non ne è disponibile alcuno.
        """

        system = platform.system()

        if system == "Darwin":
            return ["afplay", self.sound_path]

        if system == "Windows":
            return [
                "powershell",
                "-NoProfile",
                "-Command",
                "(New-Object Media.SoundPlayer "
                "'{}').PlaySync()".format(self.sound_path),
            ]

        for player in ("paplay", "aplay"):
            if shutil.which(player):
                return [player, self.sound_path]

        return None

    def _play_sound(self):
        """
        Riproduce il file audio in un processo esterno.
        """

        process = None
        command = self._player_command()

        if command is None:
            with self._lock:
                self._is_playing = False
            return

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with self._lock:
                self._process = process

                stop_requested = self._stop_requested
                enabled = self._enabled

            # Audio disattivato nel frattempo: interrompe subito.
            if (
                stop_requested
                or not enabled
            ) and process.poll() is None:
                process.terminate()

            process.wait()

        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

                self._is_playing = False
