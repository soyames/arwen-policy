from __future__ import annotations

from typing import Dict, Optional

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    _TRANSLATION_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    MarianMTModel = None  # type: ignore[assignment]
    MarianTokenizer = None  # type: ignore[assignment]
    _TRANSLATION_AVAILABLE = False


class Translator:
    """Translation fallback using MarianMT models."""

    def __init__(self, device: str = "cuda"):
        if not _TRANSLATION_AVAILABLE:
            raise RuntimeError(
                "Translation dependencies not installed. "
                "Install with: pip install torch transformers"
            )
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, MarianMTModel] = {}
        self.tokenizers: Dict[str, MarianTokenizer] = {}

    def _load_model_pair(self, src_lang: str, tgt_lang: str = "eng") -> tuple:
        """Load translation model for language pair"""
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.models[f"{src_lang}-{tgt_lang}"] = model
            self.tokenizers[f"{src_lang}-{tgt_lang}"] = tokenizer
            return model, tokenizer
        except Exception as e:
            raise ValueError(f"No translation model for {src_lang} to {tgt_lang}: {str(e)}")

    def translate(self, text: str, src_lang: str, tgt_lang: str = "eng") -> str:
        """Translate text using MarianMT"""
        if src_lang == tgt_lang:
            return text

        key = f"{src_lang}-{tgt_lang}"
        if key not in self.models:
            model, tokenizer = self._load_model_pair(src_lang, tgt_lang)
        else:
            model, tokenizer = self.models[key], self.tokenizers[key]

        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            translated = model.generate(**inputs)

        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_segments(self, segments: list, src_lang: str, tgt_lang: str = "eng") -> list:
        """Translate transcript segments while preserving timestamps"""
        translated_segments = []
        for seg in segments:
            translated_text = self.translate(seg['text'], src_lang, tgt_lang)
            translated_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': translated_text,
                'speaker': seg.get('speaker'),
                'confidence': seg.get('confidence', 0.9)
            })
        return translated_segments

# Factory function
def create_translator(device: str = "cuda") -> Translator:
    """Create translator with appropriate device."""
    return Translator(device=device)


# Public API — lazy, only if translation dependencies are available.
translator: Optional[Translator] = None
if _TRANSLATION_AVAILABLE:
    translator = create_translator()