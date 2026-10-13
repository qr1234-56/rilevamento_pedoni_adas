// ----------------------------------------------------------------------
// RIFERIMENTI AGLI ELEMENTI DELLA PAGINA
// ----------------------------------------------------------------------

// Elementi relativi alla webcam.
const webcamButton = document.getElementById("webcam-button");
const webcamContainer = document.getElementById("webcam-container");
const webcamStream = document.getElementById("webcam-stream");
const webcamStatus = document.getElementById("webcam-status");

// Elementi relativi all'analisi video.
const videoForm = document.getElementById("video-form");
const videoInput = document.getElementById("video-file");
const analysisButton = document.getElementById("analysis-button");

// Elementi relativi al report finale.
const analyzedFile = document.getElementById("analyzed-file");
const framesProcessed = document.getElementById("frames-processed");
const greenAlertFrames = document.getElementById("green-alert-frames");
const yellowAlertFrames = document.getElementById("yellow-alert-frames");
const redAlertFrames = document.getElementById("red-alert-frames");
const downloadButton = document.getElementById("download-button");


// ----------------------------------------------------------------------
// STATO DELLA WEBCAM
// ----------------------------------------------------------------------

// Tiene traccia dello stato corrente della webcam.
let webcamActive = false;


// ----------------------------------------------------------------------
// GESTIONE WEBCAM
// ----------------------------------------------------------------------

webcamButton.addEventListener("click", async () => {
    // Se la webcam è attiva, il pulsante permette di disattivarla.
    if (webcamActive) {
        try {
            // Comunica al backend che la webcam deve essere arrestata.
            // In questo modo viene fermato anche l'eventuale allarme audio.
            await fetch("/stop-webcam", {
                method: "POST",
            });
        } catch (error) {
            console.error("Errore durante l'arresto della webcam:", error);
        }

        // Nasconde il flusso video dalla pagina.
        webcamContainer.classList.add("hidden");

        // Rimuove la sorgente dello stream dal tag img.
        webcamStream.src = "";

        // Aggiorna lo stato della webcam nell'interfaccia.
        webcamStatus.textContent = "OFFLINE";

        // Aggiorna il testo e lo stile del pulsante.
        webcamButton.textContent = "Attiva webcam";
        webcamButton.classList.remove("active");

        // Aggiorna lo stato interno.
        webcamActive = false;

        return;
    }

    // ------------------------------------------------------------------
    // ATTIVAZIONE DELLA WEBCAM
    // ------------------------------------------------------------------

    try {
        // Mostra il contenitore della webcam.
        webcamContainer.classList.remove("hidden");

        // Imposta lo stream MJPEG prodotto dal backend.
        webcamStream.src = "/video_feed";

        // Aggiorna lo stato visualizzato nella dashboard.
        webcamStatus.textContent = "ONLINE";

        // Modifica il pulsante per indicare che la webcam è attiva.
        webcamButton.textContent = "Disattiva webcam";
        webcamButton.classList.add("active");

        // Aggiorna lo stato interno.
        webcamActive = true;

    } catch (error) {
        // Gestisce eventuali problemi durante l'attivazione.
        console.error("Errore durante l'attivazione della webcam:", error);

        webcamStatus.textContent = "ERRORE";
    }
});


// ----------------------------------------------------------------------
// GESTIONE UPLOAD E ANALISI VIDEO
// ----------------------------------------------------------------------

videoForm.addEventListener("submit", async (event) => {
    // Impedisce al browser di ricaricare la pagina al submit del form.
    event.preventDefault();

    // Controlla che l'utente abbia selezionato un file.
    if (!videoInput.files.length) {
        alert("Seleziona un file video.");
        return;
    }

    // Recupera il file selezionato.
    const file = videoInput.files[0];

    // Crea un oggetto FormData da inviare al backend.
    const formData = new FormData();

    // Inserisce il video nel campo "file" previsto dall'endpoint FastAPI.
    formData.append("file", file);

    // Disabilita temporaneamente il pulsante per evitare
    // l'avvio di più analisi contemporaneamente.
    analysisButton.disabled = true;
    analysisButton.textContent = "Analisi in corso...";

    try {
        // Invia il video all'endpoint /detect-video.
        const response = await fetch("/detect-video", {
            method: "POST",
            body: formData,
        });

        // Converte la risposta del backend in formato JSON.
        const result = await response.json();

        // Controlla se il backend ha segnalato un errore.
        if (result.status !== "success") {
            throw new Error(result.message || "Errore durante l'analisi.");
        }

        // ------------------------------------------------------------------
        // AGGIORNAMENTO DEL REPORT
        // ------------------------------------------------------------------

        // Mostra il nome del video analizzato.
        analyzedFile.textContent = result.input_file;

        // Mostra il numero totale di frame elaborati.
        framesProcessed.textContent = result.frames_processed;

        // Mostra il numero di frame classificati GREEN.
        greenAlertFrames.textContent = result.green_frames;

        // Mostra il numero di frame classificati YELLOW.
        yellowAlertFrames.textContent = result.yellow_frames;

        // Mostra il numero di frame classificati RED.
        redAlertFrames.textContent = result.red_frames;

        // Configura il pulsante per scaricare il video elaborato.
        downloadButton.href = `/download-video/${encodeURIComponent(
            result.output_file
        )}`;

        // Rende disponibile il pulsante di download.
        downloadButton.classList.remove("hidden");

    } catch (error) {
        // Visualizza l'errore in console e informa l'utente.
        console.error("Errore durante l'analisi:", error);
        alert(error.message || "Errore durante l'analisi del video.");

    } finally {
        // Riabilita sempre il pulsante al termine dell'operazione.
        analysisButton.disabled = false;
        analysisButton.textContent = "Avvia analisi";
    }
});

// ----------------------------------------------------------------------
// VISUALIZZAZIONE DEL NOME DEL FILE SELEZIONATO
// ----------------------------------------------------------------------

videoInput.addEventListener("change", () => {
    // Recupera il primo file selezionato dall'utente.
    const selectedFile = videoInput.files[0];

    // Individua il testo principale dell'area di caricamento.
    const uploadTitle = document.querySelector(".upload-title");

    // Se è stato selezionato un file, visualizza il suo nome.
    if (selectedFile) {
        uploadTitle.textContent = selectedFile.name;
    } else {
        // Ripristina il testo originale se il file viene rimosso.
        uploadTitle.textContent = "Seleziona un video";
    }
});
