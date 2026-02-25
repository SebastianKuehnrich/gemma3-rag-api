"""
===============================================================================
 fine_tune.py - Fine-Tuning fuer Gemma-3-1B-it
===============================================================================

 Beschreibung:
    Fine-Tuning von Gemma-3-1B-it auf deutschen Wikipedia-Daten.
    Gleicher Ansatz wie Referenz-Code (german-gpt2), aber mit:
      - pydantic-settings statt hardcoded Konstanten
      - Defensive Coding (try/except, Validierung)
      - HF_TOKEN aus .env (Gemma ist ein gated model)
      - device_map="auto" fuer optimale Hardware-Nutzung
      - wandb Integration fuer Experiment-Tracking

 Ausfuehrung:
    cd Deployment2
    ..\.venv\Scripts\python.exe fine_tune.py

 Hinweis:
    Fine-Tuning auf CPU ist sehr langsam (~Stunden).
    Empfehlung: GPU mit min. 8GB VRAM oder Google Colab.

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import logging
import math
import os
import sys
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

# .env laden bevor Settings importiert werden
load_dotenv()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import load_dataset

# ===========================================================================
#  Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
#  Konfiguration (aus .env oder Defaults)
#  Gleicher Ansatz wie Referenz, aber mit os.getenv statt Hardcoding
# ===========================================================================

MODEL_NAME    = os.getenv("GEMMA_MODEL_NAME", "google/gemma-3-1b-it")
HF_TOKEN      = os.getenv("HF_TOKEN")                   # Pflicht fuer Gemma
DATASET_NAME  = "wikimedia/wikipedia"
DATASET_LANG  = "20231101.de"                           # Deutsch wie Referenz
OUTPUT_DIR    = "./my_finetuned_gemma"
NUM_SAMPLES   = 800                                      # Wie Referenz-Code
MAX_LENGTH    = 128                                      # Wie Referenz-Code
EPOCHS        = 1                                        # Wie Referenz-Code
BATCH_SIZE    = 4                                        # Kleiner als Referenz (CPU)
SAVE_STEPS    = 200
LOGGING_STEPS = 20


def validate_config():
    """Prueft ob alle notwendigen Konfigurationen vorhanden sind."""
    if not HF_TOKEN:
        logger.error("HF_TOKEN fehlt in .env! Gemma ist ein gated model.")
        sys.exit(1)
    logger.info(f"Modell:   {MODEL_NAME}")
    logger.info(f"Output:   {OUTPUT_DIR}")
    logger.info(f"Samples:  {NUM_SAMPLES}")
    logger.info(f"Epochen:  {EPOCHS}")


def load_model_and_tokenizer():
    """
    Laedt Modell und Tokenizer mit HF Token.
    Defensive: pad_token nur setzen wenn nicht vorhanden (wie in app/).
    """
    logger.info(f"Lade Modell: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
    )

    # Pad-Token defensiv setzen (Referenz setzt es direkt, wir pruefen erst)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",       # Besser als Referenz: auto-Device-Zuweisung
        token=HF_TOKEN,
    )

    # Tokenizer-Vokabular mit Modell synchronisieren (wie Referenz)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Modell und Tokenizer geladen.")
    return model, tokenizer


def load_and_tokenize_dataset(tokenizer):
    """
    Laedt Wikipedia-Daten und tokenisiert sie.
    Gleiche Logik wie Referenz-Code.
    """
    logger.info(f"Lade Dataset: {DATASET_NAME} ({NUM_SAMPLES} Samples)...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_LANG,
        split=f"train[:{NUM_SAMPLES}]",
    )
    logger.info(f"{len(dataset)} Samples geladen.")

    def tokenize(batch):
        """
        Tokenisiert einen Batch.
        Labels = input_ids (naechster Token ist das Ziel - wie Referenz-Kommentar).
        """
        result = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        # Kein One-Hot, kein Label-Encoding - naechster Token ist das Ziel
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Tokenisiere Dataset...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,  # Nur input_ids, attention_mask, labels
    )
    tokenized.set_format("torch")
    logger.info("Tokenisierung abgeschlossen.")

    return tokenized


# ===========================================================================
#  WandbRichCallback
#  Erbt von TrainerCallback und greift in den Trainingsloop ein.
#  - on_log:        berechnet Perplexity aus dem aktuellen Loss
#  - on_step_end:   generiert alle 20 Steps Text fuer 3 Prompts
#  - on_train_end:  pusht die gesammelte Tabelle auf wandb
# ===========================================================================

SAMPLE_PROMPTS = [
    "Machine Learning ist",
    "Die Hauptstadt von Deutschland ist",
    "Ein neuronales Netz besteht aus",
]


class WandbRichCallback(TrainerCallback):
    """
    Erweiterter Callback der Perplexity und generierte Texte in wandb loggt.
    Perplexity = exp(loss) — Standard-Metrik fuer Sprachmodelle.
    Je niedriger, desto besser (Modell ist weniger 'verwirrt').
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._generation_rows = []          # Wird am Ende als Tabelle gepusht

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        """Wird nach jedem logging_steps-Intervall aufgerufen."""
        if logs and "loss" in logs:
            perplexity = math.exp(logs["loss"])
            wandb.log({"perplexity": perplexity}, step=state.global_step)
            logger.info(f"Step {state.global_step} — Loss: {logs['loss']:.4f}, Perplexity: {perplexity:.2f}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Alle 20 Steps: generiert Text fuer 3 Prompts und speichert in Tabelle."""
        if state.global_step % 20 != 0 or state.global_step == 0:
            return

        model.eval()
        with torch.no_grad():
            for prompt in SAMPLE_PROMPTS:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(model.device)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                )
                self._generation_rows.append(
                    [state.global_step, prompt, generated]
                )
        model.train()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Pusht die gesammelten Generierungen als wandb Table."""
        if not self._generation_rows:
            return

        table = wandb.Table(columns=["step", "prompt", "generated_text"])
        for row in self._generation_rows:
            table.add_data(*row)

        wandb.log({"generated_texts": table})
        logger.info(f"wandb Table mit {len(self._generation_rows)} Generierungen gepusht.")


def run_training(model, tokenizer, tokenized_dataset):
    """
    Startet das Training mit Trainer API.
    Gleiche Struktur wie Referenz, aber mit besserer Konfiguration.
    """
    run_name = f"gemma-finetuned-{MODEL_NAME.split('/')[-1]}"

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to="wandb",          # wandb wie Referenz-Code
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,              # Causal LM, nicht Masked LM
        ),
        callbacks=[WandbRichCallback(tokenizer=tokenizer)],
    )

    logger.info("Starte Training...")
    trainer.train()
    logger.info("Training abgeschlossen.")

    return trainer


def save_model(model, tokenizer):
    """Speichert Modell und Tokenizer."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Modell gespeichert in: {OUTPUT_DIR}")


# ===========================================================================
#  Main
# ===========================================================================

if __name__ == "__main__":
    try:
        # 1. Konfiguration pruefen
        validate_config()

        # 2. Modell laden
        model, tokenizer = load_model_and_tokenizer()

        # 3. Dataset laden und tokenisieren
        tokenized_dataset = load_and_tokenize_dataset(tokenizer)

        # 4. Training starten
        run_training(model, tokenizer, tokenized_dataset)

        # 5. Modell speichern
        save_model(model, tokenizer)

        logger.info("Fine-Tuning erfolgreich abgeschlossen!")

    except KeyboardInterrupt:
        logger.info("Training durch Benutzer abgebrochen.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fine-Tuning fehlgeschlagen: {e}")
        sys.exit(1)


# ===========================================================================
#  ABGABE-FRAGE (Pflichtantwort):
#
#  "Was ist der Unterschied zwischen einem Pre-Trained Model und einem
#   Fine-Tuned Model? Und wann wuerdest du in der Praxis Fine-Tuning
#   einsetzen statt ein fertiges Modell direkt zu benutzen?"
#
#  Antwort:
#
#  Ein Pre-Trained Model wurde auf einer riesigen, allgemeinen Datenmenge
#  trainiert (z.B. der gesamte Wikipedia-Dump oder Common Crawl). Es kennt
#  Sprache, Grammatik und allgemeines Weltwissen — aber keine spezifischen
#  Begriffe, Stile oder Aufgaben aus einer bestimmten Domäne.
#
#  Ein Fine-Tuned Model nimmt dieses Vorwissen als Startpunkt und wird
#  dann auf einem kleineren, domänen-spezifischen Datensatz weitertrainiert.
#  Die Gewichte des Pre-Trained Models werden dabei leicht angepasst,
#  sodass das Modell die neue Domäne "versteht", ohne das allgemeine Wissen
#  zu vergessen (Transfer Learning).
#
#  In der Praxis setze ich Fine-Tuning ein, wenn:
#  1. Das fertige Modell domänenspezifische Sprache nicht kennt
#     (z.B. medizinische Fachbegriffe, juristische Formulierungen).
#  2. Ein bestimmter Ausgabe-Stil benötigt wird
#     (z.B. immer in einem bestimmten Format antworten).
#  3. Das Modell ein spezifisches Verhalten zeigen soll, das durch
#     Prompt Engineering allein nicht zuverlässig erreichbar ist.
#
#  Ein fertiges Modell direkt benutzen reicht, wenn die Aufgabe allgemein
#  genug ist und Prompt Engineering ausreichend Kontrolle bietet —
#  Fine-Tuning kostet Rechenzeit und Daten, die man nur investiert,
#  wenn der Nutzen den Aufwand rechtfertigt.
# ===========================================================================
