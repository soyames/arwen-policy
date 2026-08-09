from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import langdetect

    _LANGDETECT_AVAILABLE = True
except ImportError:
    langdetect = None  # type: ignore[assignment]
    _LANGDETECT_AVAILABLE = False

try:
    from fasttext import load_model as _fasttext_load_model

    _FASTTEXT_AVAILABLE = True
except ImportError:
    _fasttext_load_model = None  # type: ignore[assignment]
    _FASTTEXT_AVAILABLE = False


class LanguageDetector:
    """Multilingual language detection system."""

    def __init__(self, fasttext_model_path: str = "resources/fasttext.bin"):
        """Initialize language detection models."""
        self.lang_detect = langdetect
        self._fasttext_model = None
        if _FASTTEXT_AVAILABLE and _fasttext_load_model:
            try:
                self._fasttext_model = _fasttext_load_model(fasttext_model_path)
            except Exception:
                self._fasttext_model = None

    def detect(self, text: str) -> str:
        """Detect language with fallback."""
        # Try fasttext for better accuracy on short texts
        if self._fasttext_model is not None:
            try:
                detected = self._fasttext_model.predict(text.strip())[0][0]
                return detected.replace("__label__", "").replace("_", "-")
            except Exception:
                pass
        # Fallback to langdetect if fasttext fails or unavailable
        if self.lang_detect is not None:
            try:
                return self.lang_detect.detect(text)
            except Exception:
                pass
        return "und"

    def is_supported(self, lang: str) -> bool:
        """Check if language is supported by current OCR/ASR configs"""
        supported_codes = [
            'eng', 'spa', 'fra', 'deu', 'zho', 'ara', 'rus', 'jpn', 'kor',
            'vie', 'ita', 'por', 'hrv', 'nld', 'ron'
        ]
        return lang in supported_codes

    def get_ocr_language(self, lang: str) -> str:
        """Map detected language to Tesseract language codes"""
        tesseract_codes = {
            'eng': 'eng', 'spa': 'spa', 'fra': 'fra', 'deu': 'deu',
            'zho': 'chi_sim', 'ara': 'ara', 'rus': 'rus', 'jpn': 'jpn',
            'kor': 'kor', 'vie': 'vie', 'ita': 'ita', 'por': 'por',
            'hrv': 'hrv', 'nld': 'nld', 'ron': 'ron'
        }
        return tesseract_codes.get(lang, 'eng')  # Default to English

    def get_asr_language(self, lang: str) -> str:
        """Map detected language to Whisper models"""
        whisper_models = {
            'eng': 'base', 'spa': 'base', 'fra': 'base', 'deu': 'base',
            'zho': 'zh', 'ara': 'ara', 'rus': 'rus', 'jpn': 'jpn',
            'kor': 'kor', 'vie': 'vie', 'ita': 'ita', 'por': 'por',
            'hrv': 'srp', 'nld': 'nl', 'ron': 'ron'
        }
        return whisper_models.get(lang, 'base')  # Base model for unsupported languages

# Public API — lazy instantiation.
language_detector: Optional[LanguageDetector] = None
if _LANGDETECT_AVAILABLE or _FASTTEXT_AVAILABLE:
    language_detector = LanguageDetector()