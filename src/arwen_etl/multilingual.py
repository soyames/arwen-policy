from __future__ import annotations

import langdetect
from fasttext import load_model
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

class LanguageDetector:
    """Multilingual language detection system"""

    def __init__(self, fasttext_model_path: str = "resources/fasttext.bin"):
        """Initialize language detection models"""
        self.lang_detect = langdetect
        self.fasttext_model = load_model(fasttext_model_path)

    def detect(self, text: str) -> str:
        """Detect language with fallback"""
        try:
            # Use fasttext for better accuracy on short texts
            detected = self.fasttext_model.predict(text.strip())[0][0]
            return detected.replace('_', '-')  # Convert fasttext format to BCP 47
        except:
            # Fallback to langdetect if fasttext fails
            return self.lang_detect.detect(text)

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

# Public API
language_detector = LanguageDetector()