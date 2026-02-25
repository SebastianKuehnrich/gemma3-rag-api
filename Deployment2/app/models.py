"""
===============================================================================
 models.py - Pydantic v2 Request/Response Models
===============================================================================

 Beschreibung:
    Type-safe Models fuer alle API Requests und Responses.
    Verwendet Pydantic v2 Syntax (field_validator, ConfigDict).

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


# ===========================================================================
#  Request Models
# ===========================================================================

class QueryRequest(BaseModel):
    """Request Model fuer RAG Query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Die Benutzer-Frage.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Anzahl Kontext-Dokumente fuer Retrieval.",
    )
    use_rag: bool = Field(
        default=True,
        description="True = RAG mit Kontext, False = nur Modell-Antwort.",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Stellt sicher, dass die Query nicht nur Whitespace ist."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query darf nicht leer oder nur Whitespace sein.")
        return stripped


# ===========================================================================
#  Response Models
# ===========================================================================

class ContextDocument(BaseModel):
    """Ein einzelnes Kontext-Dokument aus dem Retrieval."""

    content: str = Field(..., description="Dokument-Inhalt.")
    metadata: dict = Field(default_factory=dict, description="Metadaten.")
    distance: float = Field(..., ge=0.0, description="Distanz zum Query-Vektor.")


class RAGResponse(BaseModel):
    """Response Model fuer RAG Query."""

    query: str = Field(..., description="Die Original-Frage.")
    answer: str = Field(..., description="Generierte Antwort.")
    context_used: List[ContextDocument] = Field(
        default_factory=list,
        description="Verwendete Kontext-Dokumente.",
    )
    num_context_docs: int = Field(..., ge=0, description="Anzahl Kontext-Dokumente.")
    model: str = Field(..., description="Verwendetes Modell.")
    generation_time_seconds: float = Field(
        ..., ge=0.0, description="Generierungszeit in Sekunden."
    )


class CompareRequest(BaseModel):
    """Request Model fuer Base vs Fine-Tuned Vergleich."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Der Prompt fuer den Modell-Vergleich.",
    )
    max_new_tokens: int = Field(
        default=80,
        ge=10,
        le=200,
        description="Maximale Anzahl neuer Tokens.",
    )


class CompareResponse(BaseModel):
    """Response Model fuer /compare Endpoint."""

    prompt: str = Field(..., description="Der verwendete Prompt.")
    base_model_text: str = Field(..., description="Ausgabe des Base-Modells (ohne Fine-Tuning).")
    finetuned_text: str = Field(..., description="Ausgabe des Fine-Tuned Modells.")
    base_model_name: str = Field(..., description="Name des Base-Modells.")
    finetuned_model_name: str = Field(..., description="Name des Fine-Tuned Modells.")


class HealthResponse(BaseModel):
    """Health-Check Response fuer API-Monitoring."""

    status: str = Field(..., description="'healthy' oder 'unhealthy'.")
    gemma_loaded: bool = Field(..., description="Gemma Modell geladen?")
    embeddings_loaded: bool = Field(..., description="Embedding Modell geladen?")
    db_status: str = Field(..., description="'connected' oder 'disconnected'.")
    num_documents: int = Field(..., ge=0, description="Dokumente in der Datenbank.")
    model_name: str = Field(..., description="Name des Gemma Modells.")
