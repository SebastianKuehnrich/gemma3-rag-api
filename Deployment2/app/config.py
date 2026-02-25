"""
===============================================================================
 config.py - Zentrale Konfiguration (pydantic-settings)
===============================================================================

 Beschreibung:
    Alle Einstellungen zentral verwaltet ueber pydantic-settings.
    Werte koennen per .env-Datei oder Umgebungsvariablen ueberschrieben
    werden. Defaults sind so gewaehlt, dass alles out-of-the-box laeuft.

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application Settings mit Environment-Variable Support."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    # ---- HuggingFace ----
    HF_TOKEN: Optional[str] = None

    # ---- Model Configuration ----
    GEMMA_MODEL_NAME: str = "google/gemma-3-1b-it"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Generation Parameters ----
    MAX_NEW_TOKENS: int = 50   # CPU-optimiert: 50 statt 200 fuer akzeptable Wartezeit
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50

    # ---- Database ----
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "knowledge_base"

    # ---- API Configuration ----
    API_TITLE: str = "Gemma RAG System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "RAG-basiertes Q&A System powered by Gemma-3-1B-it"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8001

    # ---- Performance ----
    USE_4BIT_QUANTIZATION: bool = True
    DEVICE: str = "auto"



# Globale Instanz - einmal laden, ueberall verwenden
settings = Settings()
