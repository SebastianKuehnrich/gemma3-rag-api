"""
===============================================================================
 init_database.py - Datenbank-Initialisierung
===============================================================================

 Beschreibung:
    Laedt sample_faq.json in die ChromaDB.
    Kombiniert Frage + Antwort als Dokument-Text fuer besseren Kontext.
    Speichert question und category als Metadaten.

 Ausfuehrung:
    cd Deployment2
    python -m scripts.init_database

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import json
import logging
import sys
from pathlib import Path

# Parent-Verzeichnis zum Path hinzufuegen (fuer app-Imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings_service import embeddings_service
from app.database import db_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_sample_data():
    """Laedt Sample-FAQ-Daten in die ChromaDB."""

    # ---- JSON Datei laden ----
    data_path = Path(__file__).parent.parent / "data" / "sample_faq.json"

    if not data_path.exists():
        logger.error(f"Sample-Daten nicht gefunden: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("documents", [])
    if not items:
        logger.error("Keine Dokumente in der JSON-Datei gefunden.")
        return

    # ---- Dokumente und Metadaten extrahieren ----
    documents = []
    metadatas = []

    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = item.get("category", "allgemein")

        # Frage + Antwort kombinieren fuer besseren Kontext
        doc_text = f"Frage: {question}\n\nAntwort: {answer}"
        documents.append(doc_text)

        metadatas.append({
            "question": question,
            "category": category,
        })

    # ---- Services initialisieren ----
    logger.info("Lade Embedding Model...")
    embeddings_service.load_model()

    logger.info("Initialisiere ChromaDB...")
    db_service.initialize()

    # ---- Dokumente hinzufuegen ----
    logger.info(f"Fuege {len(documents)} Dokumente zur Datenbank hinzu...")
    db_service.add_documents(
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        f"Datenbank initialisiert mit {db_service.count()} Dokumenten."
    )


if __name__ == "__main__":
    load_sample_data()
