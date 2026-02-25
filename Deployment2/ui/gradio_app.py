"""
===============================================================================
 gradio_app.py - Gradio Interface fuer Gemma RAG System
===============================================================================

 Beschreibung:
    Web-UI mit drei Tabs:
      1. Chat:              Frage-Antwort mit Chat-History
      2. System Info:       Health-Status aller Komponenten
      3. API Dokumentation: Endpoint-Beschreibungen

    Kommuniziert mit dem FastAPI Backend per HTTP Requests.
    Verwendet gr.Blocks Layout (Gradio 6.x kompatibel).

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import gradio as gr
import requests
from typing import List, Dict

# ===========================================================================
#  API Konfiguration (wird von app.py ggf. ueberschrieben)
# ===========================================================================

API_URL = "http://127.0.0.1:8001"


# ===========================================================================
#  API Kommunikation
# ===========================================================================

def query_api(
    message: str,
    history: List[Dict],
    top_k: int,
    use_rag: bool,
) -> str:
    """
    Sendet Query an das FastAPI Backend.

    Args:
        message: Die Benutzer-Frage.
        history: Bisherige Chat-History.
        top_k:   Anzahl Kontext-Dokumente.
        use_rag: RAG aktiviert oder nicht.

    Returns:
        Formatierte Antwort als String.
    """
    if not message or not message.strip():
        return "Bitte eine Frage eingeben."

    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": message.strip(),
                "top_k": int(top_k),
                "use_rag": bool(use_rag),
            },
            timeout=300,  # Gemma auf CPU braucht 2-5 Minuten
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Keine Antwort erhalten.")
            num_docs = data.get("num_context_docs", 0)
            gen_time = data.get("generation_time_seconds", 0)

            if num_docs > 0:
                return (
                    f"{answer}\n\n"
                    f"---\n"
                    f"Kontext: {num_docs} Dokumente | "
                    f"Zeit: {gen_time}s"
                )
            return (
                f"{answer}\n\n"
                f"---\n"
                f"Kein Kontext verwendet | Zeit: {gen_time}s"
            )

        elif response.status_code == 503:
            return "Gemma Model wird noch geladen. Bitte 1-2 Minuten warten."
        else:
            return f"Fehler: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Fehler: Timeout. Das Modell braucht zu lange."
    except requests.exceptions.ConnectionError:
        return (
            "Fehler: API nicht erreichbar. "
            "Ist der FastAPI Server gestartet? "
            f"(Erwartet auf {API_URL})"
        )
    except Exception as e:
        return f"Fehler: {e}"


def get_health_status() -> str:
    """Holt den Health-Status vom Backend."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            gemma_status = "Geladen" if data.get("gemma_loaded") else "Nicht geladen"
            embed_status = "Geladen" if data.get("embeddings_loaded") else "Nicht geladen"

            return (
                f"### System Status\n\n"
                f"| Komponente | Status |\n"
                f"|---|---|\n"
                f"| **Status** | {data.get('status', 'unbekannt')} |\n"
                f"| **Gemma Model** | {gemma_status} |\n"
                f"| **Embeddings** | {embed_status} |\n"
                f"| **Datenbank** | {data.get('db_status', 'unbekannt')} |\n"
                f"| **Dokumente** | {data.get('num_documents', 0)} |\n"
                f"| **Model** | {data.get('model_name', 'unbekannt')} |\n"
            )
        return f"API Fehler: {response.status_code}"
    except Exception as e:
        return f"Health-Check fehlgeschlagen: {e}"


# ===========================================================================
#  Gradio Blocks Interface
# ===========================================================================

with gr.Blocks(title="Gemma RAG System") as demo:

    gr.Markdown(
        "# Gemma-3-1B-it RAG System\n\n"
        "Intelligentes Frage-Antwort-System mit **Retrieval Augmented Generation**.\n\n"
        "- **Semantic Search:** ChromaDB + Sentence Transformers\n"
        "- **Text Generation:** Google Gemma-3-1B-it\n"
        "- **REST API:** FastAPI mit Swagger Docs"
    )

    # ---- Tab 1: Chat ----
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            label="Konversation",
            height=500,
        )

        msg = gr.Textbox(
            label="Deine Frage",
            placeholder="Stelle eine Frage...",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Senden", variant="primary")
            clear_btn = gr.Button("Konversation loeschen")

        with gr.Accordion("Erweiterte Einstellungen", open=False):
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Anzahl Kontext-Dokumente (top_k)",
                info="Wie viele Dokumente fuer die Antwort verwenden?",
            )
            use_rag_checkbox = gr.Checkbox(
                value=True,
                label="RAG aktivieren",
                info="Deaktivieren fuer reine Modell-Antworten ohne Kontext.",
            )

        def respond(message, chat_history, top_k, use_rag):
            """Verarbeitet Chat-Nachricht und gibt Antwort zurueck."""
            if not message or not message.strip():
                return "", chat_history

            bot_message = query_api(message, chat_history, top_k, use_rag)

            # Gradio 5.x: Dict-Format mit role/content (Standard ab 5.0)
            chat_history = chat_history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": bot_message},
            ]
            return "", chat_history

        # Event Handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot, top_k_slider, use_rag_checkbox],
            outputs=[msg, chatbot],
        )
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, top_k_slider, use_rag_checkbox],
            outputs=[msg, chatbot],
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

        gr.Examples(
            examples=[
                "Was ist Machine Learning?",
                "Erklaere den Unterschied zwischen supervised und unsupervised Learning",
                "Wie funktioniert ein neuronales Netz?",
                "Was sind die Vorteile von RAG-Systemen?",
            ],
            inputs=msg,
            label="Beispiel-Fragen",
        )

    # ---- Tab 2: System Info ----
    with gr.Tab("System Info"):
        gr.Markdown("## System-Status\n\nStatus aller Komponenten pruefen.")

        health_output = gr.Markdown()
        refresh_btn = gr.Button("Status aktualisieren", variant="secondary")

        refresh_btn.click(fn=get_health_status, outputs=health_output)
        demo.load(fn=get_health_status, outputs=health_output)

    # ---- Tab 3: API Dokumentation ----
    with gr.Tab("API Dokumentation"):
        gr.Markdown(
            "## API Endpoints\n\n"
            "### POST /query\n\n"
            "Frage das RAG-System ab.\n\n"
            "**Request:**\n"
            "```json\n"
            '{\n'
            '    "query": "Deine Frage hier",\n'
            '    "top_k": 3,\n'
            '    "use_rag": true\n'
            '}\n'
            "```\n\n"
            "### GET /health\n\n"
            "System-Status abrufen.\n\n"
            "### GET /docs\n\n"
            f"Swagger UI: [{API_URL}/docs]({API_URL}/docs)"
        )


# ===========================================================================
#  Direct Execution
# ===========================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
