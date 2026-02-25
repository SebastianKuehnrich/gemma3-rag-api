"""
===============================================================================
 embeddings_service.py - Sentence Transformers Embedding Service
===============================================================================

 Beschreibung:
    Wrapper fuer sentence-transformers/all-MiniLM-L6-v2.
    Generiert 384-dimensionale Embedding-Vektoren fuer Text.
    Wird von database.py verwendet fuer Custom Embeddings in ChromaDB.

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import logging
from typing import List, Union

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Service fuer Embedding-Generierung mit Sentence Transformers."""

    def __init__(self):
        """Initialisiert den Service (Model wird NICHT sofort geladen)."""
        self.model = None

    def load_model(self):
        """
        Laedt das Sentence Transformer Model.

        Wird separat aufgerufen, damit der Import nicht blockiert.
        """
        try:
            logger.info(
                f"Lade Embedding Model: {settings.EMBEDDING_MODEL_NAME}"
            )
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding Model erfolgreich geladen.")
        except Exception as e:
            logger.error(f"Embedding Model konnte nicht geladen werden: {e}")
            self.model = None

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """
        Generiert Embeddings fuer einen oder mehrere Texte.

        Args:
            texts:             Ein einzelner Text oder eine Liste.
            batch_size:        Batch-Groesse fuer Encoding.
            show_progress_bar: Fortschrittsanzeige ein/aus.

        Returns:
            Liste von Embedding-Vektoren (List[List[float]]).

        Raises:
            RuntimeError: Wenn Model nicht geladen ist.
        """
        if not self.is_loaded():
            raise RuntimeError("Embedding Model ist nicht geladen.")

        try:
            # Einzelnen Text in Liste wrappen
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Embedding-Generierung fehlgeschlagen: {e}")
            raise

    def is_loaded(self) -> bool:
        """Prueft ob das Embedding Model geladen ist."""
        return self.model is not None


# Globale Instanz (Model wird spaeter per load_model() geladen)
embeddings_service = EmbeddingsService()
