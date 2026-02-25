"""
===============================================================================
 app.py - HuggingFace Spaces Entry Point
===============================================================================

 Beschreibung:
    Main Entry Point fuer HuggingFace Spaces Deployment.
    Startet FastAPI Backend in einem Background-Thread und
    Gradio Frontend im Main-Thread.

    Prueft nach dem Start ob FastAPI erreichbar ist (Health-Check).
    Gibt klare Fehlermeldung wenn Port belegt ist.

    Auf HuggingFace Spaces:
      - FastAPI laeuft intern auf Port 8001
      - Gradio wird von HF Spaces auf Port 7860 exposed

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import os
import socket
import threading
import time

import requests
import uvicorn

# ===========================================================================
#  Environment fuer HuggingFace Spaces setzen
# ===========================================================================

os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")

# ===========================================================================
#  App Imports (nach Environment-Setup)
# ===========================================================================

from app.main import app as fastapi_app
from app.config import settings
import ui.gradio_app as gradio_module
from ui.gradio_app import demo as gradio_app

# API URL fuer Gradio setzen (interner Zugriff)
gradio_module.API_URL = f"http://127.0.0.1:{settings.API_PORT}"


# ===========================================================================
#  Hilfsfunktionen
# ===========================================================================

def is_port_free(port: int) -> bool:
    """Prueft ob ein Port frei ist."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_api(url: str, timeout: int = 30) -> bool:
    """
    Wartet bis die API erreichbar ist.

    Args:
        url:     Health-Check URL.
        timeout: Max. Wartezeit in Sekunden.

    Returns:
        True wenn erreichbar, False bei Timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ===========================================================================
#  FastAPI Background Thread
# ===========================================================================

def run_fastapi():
    """Startet FastAPI Server im Background Thread."""
    uvicorn.run(
        fastapi_app,
        host="127.0.0.1",
        port=settings.API_PORT,
        log_level="info",
    )


# ===========================================================================
#  Main
# ===========================================================================

if __name__ == "__main__":

    # ---- Port-Check vor dem Start ----
    if not is_port_free(settings.API_PORT):
        print(
            f"\n[FEHLER] Port {settings.API_PORT} ist bereits belegt!\n"
            f"Loesung: Beende den alten Prozess mit:\n"
            f"  cmd /c \"for /f 'tokens=5' %a in ('netstat -aon ^| findstr :{settings.API_PORT}') "
            f"do taskkill /F /PID %a\"\n"
        )
        raise SystemExit(1)

    # ---- FastAPI in Background Thread starten ----
    print(f"Starte FastAPI auf Port {settings.API_PORT}...")
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()

    # ---- Auf API warten (max. 30s) ----
    health_url = f"http://127.0.0.1:{settings.API_PORT}/health"
    print(f"Warte auf FastAPI ({health_url})...")

    # Gemma braucht ~140s zum Laden - timeout grosszuegig setzen
    if wait_for_api(health_url, timeout=180):
        print("FastAPI ist bereit!")
    else:
        print("[WARNUNG] FastAPI antwortet nicht - Gradio startet trotzdem.")

    # ---- Gradio starten (blockiert - Main Thread) ----
    print("Starte Gradio UI auf Port 7860...")
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
