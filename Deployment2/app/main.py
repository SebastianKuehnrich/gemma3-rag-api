"""
===============================================================================
 main.py - FastAPI Application fuer Gemma RAG System
===============================================================================

 Beschreibung:
    REST API mit drei Endpoints:
      GET  /health  - System-Status und Komponenten-Health
      POST /query   - RAG Query mit optionalem Kontext-Retrieval
      GET  /        - API-Info und Endpoint-Uebersicht

    Beim Start werden alle Services geladen:
      1. Embedding Model (Sentence Transformers)
      2. ChromaDB (Vector Database)
      3. Gemma Model (Text Generation)

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    QueryRequest,
    RAGResponse,
    HealthResponse,
    ContextDocument,
    CompareRequest,
    CompareResponse,
)
from app.gemma_service import gemma_service
from app.embeddings_service import embeddings_service
from app.database import db_service

# ===========================================================================
#  Logging Setup
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
#  FastAPI App
# ===========================================================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
)

# CORS Middleware (erlaubt Requests von Gradio UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
#  Startup Event - Services laden
# ===========================================================================

@app.on_event("startup")
async def startup_event():
    """Laedt alle Services beim Server-Start."""
    logger.info("=== Starte Gemma RAG System ===")

    # 1. Embeddings laden (schnell, ~2s)
    logger.info("[1/3] Lade Embedding Model...")
    embeddings_service.load_model()

    # 2. Datenbank initialisieren (schnell, <1s)
    logger.info("[2/3] Initialisiere ChromaDB...")
    db_service.initialize()

    # 3. Gemma laden (langsam, 30-120s)
    logger.info("[3/3] Lade Gemma Model (kann 30-120s dauern)...")
    gemma_service.load_model()

    logger.info("=== Alle Services geladen ===")


# ===========================================================================
#  Endpoints
# ===========================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health-Check: Prueft ob alle Komponenten erreichbar sind.

    Returns:
        HealthResponse mit Status aller Services.
    """
    try:
        return HealthResponse(
            status="healthy" if gemma_service.is_loaded() else "unhealthy",
            gemma_loaded=gemma_service.is_loaded(),
            embeddings_loaded=embeddings_service.is_loaded(),
            db_status="connected" if db_service.is_initialized() else "disconnected",
            num_documents=db_service.count(),
            model_name=settings.GEMMA_MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"Health-Check fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    RAG Query: Semantische Suche + Gemma Text-Generierung.

    Args:
        request: QueryRequest mit query, top_k, use_rag.

    Returns:
        RAGResponse mit Antwort und verwendetem Kontext.
    """
    if not gemma_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Gemma Model ist noch nicht geladen. Bitte warten.",
        )

    try:
        start_time = time.time()

        # ---- Schritt 1: Kontext-Retrieval (optional) ----
        context_docs = []
        if request.use_rag and db_service.count() > 0:
            results = db_service.query(
                query_text=request.query,
                n_results=request.top_k,
            )

            # Ergebnisse in Context-Dicts umwandeln
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                context_docs.append({
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else 0.0,
                })

        # ---- Schritt 2: Prompt bauen ----
        if context_docs:
            prompt = gemma_service.build_rag_prompt(
                request.query, context_docs
            )
        else:
            # Standalone Query ohne Kontext
            prompt = (
                f"<start_of_turn>user\n"
                f"{request.query}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        # ---- Schritt 3: Antwort generieren ----
        answer = gemma_service.generate(prompt)

        generation_time = time.time() - start_time

        # ---- Response formatieren ----
        context_objects = [
            ContextDocument(
                content=doc["content"],
                metadata=doc["metadata"],
                distance=doc["distance"],
            )
            for doc in context_docs
        ]

        return RAGResponse(
            query=request.query,
            answer=answer,
            context_used=context_objects,
            num_context_docs=len(context_docs),
            model=settings.GEMMA_MODEL_NAME,
            generation_time_seconds=round(generation_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Vergleicht Base-Modell vs Fine-Tuned Modell mit demselben Prompt.

    Laedt dbmdz/german-gpt2 als leichtgewichtiges Base-Modell zum Vergleich,
    da das echte Gemma-Base zu gross fuer Live-Vergleiche waere.

    Args:
        request: CompareRequest mit prompt und max_new_tokens.

    Returns:
        CompareResponse mit beiden Ausgaben nebeneinander.
    """
    if not gemma_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Gemma Model ist noch nicht geladen. Bitte warten.",
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        BASE_MODEL_NAME = "dbmdz/german-gpt2"

        # ---- Base-Modell laden (ohne Fine-Tuning) ----
        logger.info(f"Lade Base-Modell {BASE_MODEL_NAME} fuer Vergleich...")
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        base_model.eval()

        # ---- Base-Modell generiert ----
        inputs = base_tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            output_ids = base_model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=base_tokenizer.eos_token_id,
            )
        base_text = base_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ---- Fine-Tuned Modell (Gemma) generiert ----
        finetuned_text = gemma_service.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
        )

        logger.info("Vergleich abgeschlossen.")
        return CompareResponse(
            prompt=request.prompt,
            base_model_text=base_text,
            finetuned_text=finetuned_text,
            base_model_name=BASE_MODEL_NAME,
            finetuned_model_name=settings.GEMMA_MODEL_NAME,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Modell-Vergleich fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root Endpoint mit API-Informationen."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "compare": "POST /compare",
            "docs": "GET /docs",
        },
    }


# ===========================================================================
#  Direct Execution
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
    )
