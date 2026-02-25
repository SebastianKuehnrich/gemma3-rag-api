# gemma3-rag-api

Intelligentes Frage-Antwort-System mit Retrieval Augmented Generation (RAG), gebaut mit Google Gemma-3-1B-it, ChromaDB, FastAPI und Gradio.

## Tech Stack

| Komponente | Technologie |
|---|---|
| Text Generation | Google Gemma-3-1B-it (4-bit quantized) |
| Semantic Search | ChromaDB + Sentence Transformers |
| Backend API | FastAPI + Pydantic v2 |
| Web UI | Gradio (gr.Blocks) |
| Experiment Tracking | wandb |
| Fine-Tuning | HuggingFace Trainer + Causal LM |

## Architektur

```
User
  |
  v
Gradio UI (Port 7860)
  |
  v
FastAPI Backend (Port 8001)
  |
  +---> ChromaDB (Semantic Search)
  |         |
  |         v
  |     Sentence Transformers (all-MiniLM-L6-v2)
  |
  +---> Gemma-3-1B-it (Text Generation, 4-bit)
```

## API Endpoints

| Method | Endpoint | Beschreibung |
|---|---|---|
| GET | `/health` | System-Status aller Komponenten |
| POST | `/query` | RAG Query: Semantic Search + Text Generation |
| POST | `/compare` | Base-Modell vs Fine-Tuned Modell Vergleich |
| GET | `/docs` | Swagger UI |

## Setup

### Voraussetzungen

- Python 3.10+
- HuggingFace Account mit Zugriff auf [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- GPU empfohlen (8GB+ VRAM), oder Google Colab (siehe unten)

### Installation

```bash
git clone https://github.com/SebastianKuehnrich/gemma3-rag-api.git
cd gemma3-rag-api/Deployment2

pip install -r requirements.txt
```

### Konfiguration

```bash
cp .env.example .env
# .env oeffnen und HF_TOKEN eintragen
```

`.env` Pflichtfeld:
```
HF_TOKEN=your_huggingface_token_here
```

### Datenbank initialisieren

```bash
python -m scripts.init_database
```

### Starten

```bash
python app.py
```

- Gradio UI: http://localhost:7860
- Swagger UI: http://localhost:8001/docs

## Google Colab (empfohlen ohne GPU)

Das Notebook `colab_run.ipynb` startet das komplette System auf einer kostenlosen T4 GPU:

1. [colab.research.google.com](https://colab.research.google.com) öffnen
2. `Laufzeit` → `Laufzeittyp ändern` → **T4 GPU**
3. `colab_run.ipynb` hochladen
4. In Zelle 3: GitHub URL eintragen
5. In Zelle 4: HF Token eintragen
6. Alle Zellen ausführen → öffentliche Gradio URL erscheint

## Fine-Tuning

Fine-Tuning von Gemma auf deutschen Wikipedia-Daten mit wandb Tracking:

```bash
# WANDB_API_KEY in .env eintragen
python fine_tune.py
```

Das Training loggt automatisch:
- Loss-Kurve
- Perplexity pro Step
- Generierte Texte alle 20 Steps (als wandb Table)

Trainiertes Modell wird in `./my_finetuned_gemma/` gespeichert und beim nächsten Start automatisch geladen.

## Projekt-Struktur

```
Deployment2/
├── app/
│   ├── main.py              # FastAPI Endpoints
│   ├── gemma_service.py     # Gemma Model Service
│   ├── embeddings_service.py # Sentence Transformers
│   ├── database.py          # ChromaDB Service
│   ├── models.py            # Pydantic Request/Response Models
│   └── config.py            # Zentrale Konfiguration
├── ui/
│   └── gradio_app.py        # Gradio Web Interface
├── scripts/
│   └── init_database.py     # Datenbank-Initialisierung
├── data/
│   └── sample_faq.json      # ML/AI FAQ Datensatz (10 Eintraege)
├── fine_tune.py             # Fine-Tuning Script mit WandbRichCallback
├── app.py                   # Entry Point (FastAPI + Gradio)
├── colab_run.ipynb          # Google Colab Notebook
├── requirements.txt
└── .env.example
```

## Keywords

`Fine-Tuning` `Causal Language Modeling` `HuggingFace Transformers` `FastAPI` `Model Serving` `REST API` `Pydantic` `Gradio` `ML Demo` `Rapid Prototyping` `wandb` `Experiment Tracking` `MLOps` `German NLP` `ChromaDB` `RAG` `Retrieval Augmented Generation`

---

*Sebastian Kühnrich — 2026*
