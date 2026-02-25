"""
===============================================================================
 Deployment2/app - Gemma-3-1B-it RAG System
===============================================================================

 Backend-Package fuer das RAG-System mit:
   - config.py:            Zentrale Konfiguration (pydantic-settings)
   - models.py:            Pydantic v2 Request/Response Models
   - gemma_service.py:     Gemma Model Service (4-bit Quantization)
   - embeddings_service.py: Sentence Transformers Embeddings
   - database.py:          ChromaDB Vector Database
   - main.py:              FastAPI Application

 Autor:   Sebastian
 Datum:   2026-02-25
 Version: 1.0.0
===============================================================================
"""
