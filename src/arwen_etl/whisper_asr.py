from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import soundfile as sf
    import torch
    import whisper

    _ASR_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    sf = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    whisper = None  # type: ignore[assignment]
    _ASR_AVAILABLE = False


class WhisperASRProcessor:
    """GPU-accelerated ASR pipeline using Whisper via Hugging Face transformers."""

    def __init__(self, model_size: str = "base", device: Optional[str] = None, compute_type: str = "float16"):
        """Initialize with model configuration.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu). If None, auto-detect.
            compute_type: Compute type for inference (float16, int8, etc.)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_size = model_size
        self.compute_type = compute_type

        # Load Whisper model
        self.model = whisper.load_model(model_size, device=self.device)
        self.sample_rate = 16000  # Whisper expects 16kHz audio

    def _normalize_audio(self, audio_path: str) -> np.ndarray:
        """Normalize audio to 16kHz mono PCM WAV.

        Args:
            audio_path: Path to audio file

        Returns:
            Normalized audio as numpy array
        """
        # Load audio with soundfile (handles WAV, FLAC, etc.)
        data, samplerate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(data.shape) > 1 and data.shape[1] == 2:
            data = np.mean(data, axis=1)

        # Resample to 16kHz if needed
        if samplerate != self.sample_rate:
            # Use scipy or librosa for resampling if available, otherwise simple interpolation
            # For simplicity, we'll use numpy interpolation (not ideal but functional)
            import scipy.signal
            num_samples = int(len(data) * self.sample_rate / samplerate)
            data = scipy.signal.resample(data, num_samples)

        # Ensure float32 in range [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0

        return data.astype(np.float32)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "en", "es"). If None, auto-detect.

        Returns:
            Dictionary with transcription, segments, and metadata
        """
        # Normalize audio
        audio_data = self._normalize_audio(audio_path)

        # Transcribe with Whisper
        result = self.model.transcribe(
            audio_data,
            language=language,
            task="transcribe",
            fp16=(self.device == "cuda"),  # Use FP16 on GPU for speed
        )

        # Format output with timestamps and confidence
        transcription_segments = []
        for segment in result.get("segments", []):
            transcription_segments.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0.0),  # Whisper doesn't provide confidence directly
                "words": [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": w.get("probability", 0.0)
                    }
                    for w in segment.get("words", [])
                ] if "words" in segment else []
            })

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": transcription_segments,
            "duration": result.get("duration", 0.0),
            "model": self.model_size,
            "device": self.device,
        }

    def transcribe_with_timestamps(self, audio_path: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get transcribe with word-level timestamps.

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            List of word-level timestamped segments
        """
        result = self.transcribe(audio_path, language)
        return result.get("segments", [])


# Factory function for easy instantiation
def create_asr_processor(model_size: str = "base", device: Optional[str] = None) -> WhisperASRProcessor:
    """Create ASR processor with sensible defaults."""
    return WhisperASRProcessor(model_size=model_size, device=device)


# Public API - default processor (lazy, only if dependencies available)
asr_processor: Optional[WhisperASRProcessor] = None
if _ASR_AVAILABLE:
    asr_processor = create_asr_processor()