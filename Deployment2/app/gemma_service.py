"""
===============================================================================
 gemma_service.py - Gemma-3-1B-it Model Service
===============================================================================

 Beschreibung:
    Laedt Google Gemma-3-1B-it mit optionaler 4-bit Quantization
    fuer Memory-Effizienz. Bietet generate() fuer Text-Generierung
    und build_rag_prompt() fuer RAG-formatierte Prompts.

    4-bit Quantization reduziert den RAM-Bedarf von ~4GB auf ~1.5GB.

 Autor:   Sebastian
 Datum:   2026-02-25
===============================================================================
"""

import logging
import time
from typing import Optional, List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from app.config import settings

logger = logging.getLogger(__name__)


class GemmaService:
    """
    Service fuer Gemma Model Inference.

    Zwei Modi (gesteuert via USE_HF_INFERENCE_API in config):
      - API-Modus:   HTTP-Calls an HuggingFace Inference API (keine GPU noetig)
      - Lokal-Modus: Modell wird in RAM/VRAM geladen (braucht GPU fuer Echtzeit)
    """

    def __init__(self):
        """Initialisiert den Service (Model wird NICHT sofort geladen)."""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None

    def _resolve_model_path(self) -> str:
        """
        Bestimmt den Model-Pfad nach Lehrer-Workflow:
          1. Lokales Fine-Tuned Modell (./my_finetuned_gemma) - Prioritaet
          2. HuggingFace Modell als Fallback

        Returns:
            Pfad oder HuggingFace Model-Name.
        """
        import os
        local_path = "./my_finetuned_gemma"
        if os.path.isdir(local_path):
            logger.info(f"Lokales Fine-Tuned Modell gefunden: {local_path}")
            return local_path
        logger.info(f"Kein lokales Modell - lade von HuggingFace: {settings.GEMMA_MODEL_NAME}")
        return settings.GEMMA_MODEL_NAME

    def load_model(self):
        """
        Laedt Gemma Model mit Optimierungen.

        Workflow (wie Lehrer-Code):
          1. Prueft ob lokales Fine-Tuned Modell vorhanden (fine_tune.py Output)
          2. Fallback: HuggingFace Download mit HF_TOKEN
          3. Bei Fehler: self.model = None (graceful degradation wie Lehrer)

        Wird separat aufgerufen, damit der Import nicht blockiert.
        """
        try:
            model_path = self._resolve_model_path()
            logger.info(f"Lade Gemma Model: {model_path}")
            start_time = time.time()

            # Token nur bei HF-Download noetig (lokal kein Token noetig)
            token = settings.HF_TOKEN if model_path == settings.GEMMA_MODEL_NAME else None

            # ---- 4-bit Quantization fuer Memory-Effizienz ----
            if settings.USE_4BIT_QUANTIZATION:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map=settings.DEVICE,
                    trust_remote_code=True,
                    token=token,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=settings.DEVICE,
                    trust_remote_code=True,
                    token=token,
                )

            # ---- Tokenizer laden ----
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=token,
            )

            # Pad-Token setzen falls nicht vorhanden (defensiver als Lehrer)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.device = next(self.model.parameters()).device
            self.model_path = model_path

            load_time = time.time() - start_time
            logger.info(
                f"Gemma Model geladen in {load_time:.2f}s auf {self.device}"
            )

        except Exception as e:
            # Wie Lehrer-Code: model = None bei Fehler + hilfreiche Meldung
            logger.error(
                f"Kein Modell gefunden. Zuerst fine_tune.py ausfuehren "
                f"oder HF_TOKEN setzen! Fehler: {e}"
            )
            self.model = None
            self.tokenizer = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generiert Text mit Gemma.

        Args:
            prompt:         Der Input-Prompt.
            max_new_tokens: Max. Tokens (default aus config).
            temperature:    Sampling Temperature (default aus config).
            top_p:          Nucleus Sampling (default aus config).
            top_k:          Top-K Sampling (default aus config).

        Returns:
            Generierter Text (nur der neue Teil, ohne Prompt).

        Raises:
            RuntimeError: Wenn Model nicht geladen ist.
        """
        if not self.is_loaded():
            raise RuntimeError("Gemma Model ist nicht geladen.")

        try:
            # Defaults aus config verwenden
            max_new_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
            temperature = temperature or settings.TEMPERATURE
            top_p = top_p or settings.TOP_P
            top_k = top_k or settings.TOP_K

            # Tokenisieren
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            # Generieren (ohne Gradient-Berechnung)
            # CPU-Optimierung: Greedy Decoding (do_sample=False) ist 3-5x
            # schneller als Sampling, da kein temperature/top_p/top_k noetig.
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Nur den generierten Teil decodieren (ohne Prompt)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Text-Generierung fehlgeschlagen: {e}")
            raise

    def build_rag_prompt(
        self, query: str, context_docs: List[Dict[str, str]]
    ) -> str:
        """
        Baut einen RAG-Prompt mit Kontext-Dokumenten.

        Args:
            query:        Die Benutzer-Frage.
            context_docs: Liste von Dicts mit 'content' Key.

        Returns:
            Formatierter Prompt im Gemma Chat-Format.
        """
        # Kontext-Dokumente formatieren
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get("content", "")
            context_parts.append(f"[Dokument {i + 1}]\n{content}")

        context_text = "\n\n".join(context_parts)

        # Gemma Chat-Format
        prompt = (
            f"<start_of_turn>user\n"
            f"Du bist ein hilfreicher Assistent. Beantworte die Frage "
            f"basierend auf dem folgenden Kontext.\n\n"
            f"Kontext:\n{context_text}\n\n"
            f"Frage: {query}\n\n"
            f"Gib eine praezise Antwort basierend auf dem Kontext. "
            f"Falls der Kontext die Frage nicht beantwortet, "
            f"sage das ehrlich.<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        return prompt

    def is_loaded(self) -> bool:
        """Prueft ob Model und Tokenizer geladen sind."""
        return self.model is not None and self.tokenizer is not None


# Globale Instanz (Model wird spaeter per load_model() geladen)
gemma_service = GemmaService()
