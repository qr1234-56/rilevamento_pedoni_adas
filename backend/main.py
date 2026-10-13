from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import router


# Crea l'applicazione FastAPI principale.
app = FastAPI(
    title="Pedestrian Detection",
    description="Sistema di rilevamento pedoni per applicazioni ADAS.",
    version="1.0.0",
)


# ----------------------------------------------------------------------
# CONFIGURAZIONE CORS
# ----------------------------------------------------------------------

# Permette al frontend di comunicare con il backend.
# In un'applicazione locale vengono consentite le richieste
# provenienti dalle diverse origini utilizzate dal browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# CARTELLE DELL'APPLICAZIONE
# ----------------------------------------------------------------------

# Definisce le cartelle utilizzate per file statici, upload e output.
STATIC_DIR = Path("frontend/static")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# Crea le cartelle se non sono ancora presenti.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# FILE STATICI E RISORSE
# ----------------------------------------------------------------------

# Rende disponibili al browser i file CSS e JavaScript
# contenuti nella cartella frontend/static.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# Rende accessibili i video caricati e quelli elaborati dal backend.
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# ----------------------------------------------------------------------
# ROUTES DELL'APPLICAZIONE
# ----------------------------------------------------------------------

# Collega all'applicazione tutti gli endpoint definiti
# nel modulo backend.routes.
app.include_router(router)


# ----------------------------------------------------------------------
# AVVIO DELL'APPLICAZIONE
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Avvia il server web in modalità locale.
    # host 0.0.0.0 permette di raggiungere il server dalla rete locale.
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
