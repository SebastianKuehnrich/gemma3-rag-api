"""
===============================================================================
 database.py - ChromaDB Vector Database Service
===============================================================================

 Beschreibung:
    ChromaDB Service mit Custom Embeddings (Sentence Transformers).
    Verwendet PersistentClient fuer lokale Entwicklung.
    Embeddings werden NICHT von ChromaDB auto-generiert, sondern
    explizit ueber embeddings_service erzeugt.

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings_service import embeddings_service

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service fuer ChromaDB Operationen mit Custom Embeddings."""

    def __init__(self):
        """Initialisiert den Service (DB wird NICHT sofort verbunden)."""
        self.client = None
        self.collection = None

    def initialize(self):
        """
        Initialisiert ChromaDB Client und Collection.

        Wird separat aufgerufen, damit der Import nicht blockiert.
        Setzt anonymized_telemetry=False fuer Datenschutz.
        """
        try:
            logger.info("Initialisiere ChromaDB...")

            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                metadata={"description": "Knowledge Base fuer RAG System"},
            )

            doc_count = self.collection.count()
            logger.info(f"ChromaDB initialisiert mit {doc_count} Dokumenten.")

        except Exception as e:
            logger.error(f"ChromaDB Initialisierung fehlgeschlagen: {e}")
            self.client = None
            self.collection = None

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Fuegt Dokumente mit Custom Embeddings zur Collection hinzu.

        Args:
            documents: Liste von Dokument-Texten.
            metadatas: Optionale Metadaten pro Dokument.
            ids:       Optionale IDs (werden sonst auto-generiert).

        Raises:
            RuntimeError: Wenn DB nicht initialisiert ist.
        """
        if not self.is_initialized():
            raise RuntimeError("Datenbank ist nicht initialisiert.")

        try:
            # Custom Embeddings generieren
            embeddings = embeddings_service.encode(documents)

            # IDs auto-generieren falls nicht angegeben
            if ids is None:
                start_id = self.collection.count()
                ids = [f"doc_{start_id + i}" for i in range(len(documents))]

            # Zur Collection hinzufuegen
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"{len(documents)} Dokumente hinzugefuegt.")

        except Exception as e:
            logger.error(f"Dokumente hinzufuegen fehlgeschlagen: {e}")
            raise

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantische Suche mit Custom Embeddings.

        Args:
            query_text: Die Suchanfrage.
            n_results:  Anzahl Ergebnisse.
            where:      Optionaler Metadaten-Filter.

        Returns:
            ChromaDB Query-Results Dict.

        Raises:
            RuntimeError: Wenn DB nicht initialisiert ist.
        """
        if not self.is_initialized():
            raise RuntimeError("Datenbank ist nicht initialisiert.")

        try:
            # Custom Query-Embedding generieren
            query_embedding = embeddings_service.encode(query_text)

            # n_results darf nicht groesser sein als Anzahl Dokumente
            max_results = min(n_results, self.collection.count())
            if max_results < 1:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            results = self.collection.query(
                query_embeddings=[query_embedding[0]],
                n_results=max_results,
                where=where,
            )

            return results

        except Exception as e:
            logger.error(f"Query fehlgeschlagen: {e}")
            raise

    def count(self) -> int:
        """Gibt die Anzahl Dokumente in der Collection zurueck."""
        if not self.is_initialized():
            return 0
        return self.collection.count()

    def is_initialized(self) -> bool:
        """Prueft ob Client und Collection initialisiert sind."""
        return self.client is not None and self.collection is not None


# Globale Instanz (DB wird spaeter per initialize() verbunden)
db_service = DatabaseService()
