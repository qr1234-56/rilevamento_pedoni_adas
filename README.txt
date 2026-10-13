RILEVAMENTO PEDONI PER SISTEMI ADAS


Prototipo di sistema di assistenza alla guida che analizza il video di una
telecamera rivolta in avanti e segnala i pedoni presenti davanti al veicolo.
Per ciascun pedone stima la distanza e lo scostamento laterale rispetto alla
traiettoria e assegna un livello di allerta: verde, giallo o rosso. Il
livello rosso è accompagnato da un avviso acustico. Il sistema funziona con
la webcam del computer oppure su un file video caricato dall'interfaccia web.

Il rilevamento usa YOLOv8n, con i pesi preaddestrati sul dataset COCO, e il
tracker ByteTrack. La distanza è stimata con il modello pinhole, assumendo
un'altezza del pedone di 1,70 m; per le telecamere non calibrate la
lunghezza focale è ricavata da un campo visivo orizzontale di 60 gradi.


REQUISITI

- Python 3.12
- FFmpeg, usato per inserire l'avviso acustico nel video elaborato
  (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg;
  Windows: si scarica da https://ffmpeg.org e si aggiunge al PATH)
- una connessione a Internet al primo avvio, per scaricare i pesi yolov8n.pt


INSTALLAZIONE

Dalla cartella del progetto:

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Su Windows l'ambiente virtuale si crea con "py -3.12 -m venv .venv" e si
attiva con ".venv\Scripts\activate".

Il file requirements-lock.txt contiene le versioni esatte delle librerie con
cui il progetto è stato provato: si può usare al posto di requirements.txt
se con le versioni più recenti qualcosa non funziona.


AVVIO

    uvicorn backend.main:app --port 8000

e poi si apre http://localhost:8000 nel browser. Al primo avvio viene
scaricato yolov8n.pt nella cartella del progetto.

L'interfaccia ha due modalità:

- Telecamera: analizza in tempo reale il flusso della webcam;
- Analisi video: si carica un file .mp4, .avi o .mov; al termine vengono
  mostrati i fotogrammi verdi, gialli e rossi e si può scaricare il video
  elaborato, con l'avviso acustico. Le statistiche dell'analisi vengono
  salvate in outputs/<nome del video>_statistics.csv.

Le verifiche della logica, che non richiedono il modello né la webcam, si
lanciano con:

    python tests/test_detection_logic.py


CARTELLE

    backend/     server web (FastAPI)
    detection/   rilevamento, tracciamento, stima della distanza e allerta
    frontend/    interfaccia web
    sounds/      suono dell'avviso acustico
    tests/       verifiche della logica
    tools/       script di valutazione
    modelli/     pesi di YOLOv8n riaddestrati sul dataset KITTI
    esempi/      immagini di esempio del funzionamento


SCRIPT DI VALUTAZIONE E ADDESTRAMENTO

Si lanciano dalla cartella del progetto e salvano i risultati in results/.
I dataset non sono inclusi.

    python prepare_kitti.py
        converte il dataset KITTI nel formato di Ultralytics

    python train_kitti.py
        riaddestra YOLOv8n sul dataset convertito

    python tools/valuta_distanza_kitti.py --immagini 356 --confidenza 0.25
        rilevamento, distanza e posizione laterale su KITTI, confrontati
        con le annotazioni 3D

    python tools/valuta_iseauto.py
        rilevamento e distanza sul dataset iseAuto, confrontati con le
        maschere annotate e con il lidar

    python tools/analizza_sequenza_kitti.py --sequenza ... --calibrazione ...
        sistema completo su una sequenza grezza di KITTI (cartella della
        sequenza e dei file di calibrazione), confrontato con il lidar e
        con il GPS del veicolo

    python tools/misura_webcam.py
        tempo di elaborazione per fotogramma con la webcam

Percorsi predefiniti dei dati, relativi alla cartella del progetto:

- immagini di KITTI: dataset/kitti/raw/data_object_image_2/training/image_2
- annotazioni di KITTI: dataset/kitti/raw/data_object_label_2/training/label_2
  (in valuta_distanza_kitti.py si cambia con --etichette, in
  prepare_kitti.py con la costante LABEL_SOURCE)
- dataset iseAuto: dataset/iseauto (si cambia con --radice)

Le due cartelle di KITTI corrispondono agli archivi data_object_image_2.zip
e data_object_label_2.zip, scaricabili dal sito del dataset
(https://www.cvlibs.net/datasets/kitti/) dopo la registrazione.

valuta_distanza_kitti.py e valuta_iseauto.py usano i pesi generici
yolov8n.pt; con --modello modelli/yolov8n_kitti.pt usano quelli riaddestrati.


CREDITI E LICENZE

- Il progetto ha preso avvio da PedestrianGuard-AI
  (https://github.com/dhairya-0001/PedestrianGuard-AI).
- YOLOv8 e la libreria Ultralytics sono distribuiti con licenza AGPL-3.0.
- I dati di KITTI sono distribuiti con licenza CC BY-NC-SA 3.0, che ne
  consente l'uso solo per scopi non commerciali; i pesi in modelli/ sono
  stati ottenuti addestrando il modello su questi dati.
- Il dataset iseAuto è descritto in J. Gu, M. Bellone, R. Sell, A. Lind,
  "Object Segmentation for Autonomous Driving Using iseAuto Data",
  Electronics, 11 (2022), art. 1119.
- Le immagini in esempi/addestramento/ provengono dal dataset KITTI
  (A. Geiger, P. Lenz, R. Urtasun, "Are We Ready for Autonomous Driving?
  The KITTI Vision Benchmark Suite", CVPR 2012) e sono soggette alla stessa
  licenza CC BY-NC-SA 3.0. Quelle in esempi/inferenza/ sono fotogrammi di
  un video Pexels, utilizzabile liberamente secondo la licenza Pexels.
- Il codice di questo progetto è distribuito con licenza AGPL-3.0 (file
  LICENSE), la stessa di Ultralytics.
