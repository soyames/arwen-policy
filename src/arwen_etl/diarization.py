from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline as _PyannotePipeline

    _DIARIZATION_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
    _PyannotePipeline = None  # type: ignore[assignment]
    _DIARIZATION_AVAILABLE = False


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 use_auth_token: Optional[str] = None,
                 device: str = "cuda"):
        """Initialize diarization pipeline."""
        if not _DIARIZATION_AVAILABLE:
            raise RuntimeError(
                "Diarization dependencies not installed. "
                "Install with: pip install torch pyannote.audio"
            )
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipeline = _PyannotePipeline.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        ).to(torch.device(self.device))

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Generate speaker segments with timestamps"""
        # Load and process audio
        diarization = self.pipeline(audio_path)

        # Convert to structured format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })

        return segments

    def diarize_from_array(self, waveform: torch.Tensor, sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """Diarize from torch tensor"""
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })

        return segments

    def assign_speakers_to_transcript(self,
                                       transcript_segments: List[Dict[str, Any]],
                                       diarization_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign speaker IDs to transcript segments based on time overlap"""
        for tseg in transcript_segments:
            t_start, t_end = tseg['start'], tseg['end']

            # Find overlapping diarization segment
            best_overlap = 0
            assigned_speaker = None

            for dseg in diarization_segments:
                d_start, d_end = dseg['start'], dseg['end']
                overlap = max(0, min(t_end, d_end) - max(t_start, d_start))

                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_speaker = dseg['speaker']

            tseg['speaker'] = assigned_speaker

        return transcript_segments


# Factory function
def create_diarizer(device: str = "cuda") -> SpeakerDiarizer:
    """Create diarizer with appropriate device."""
    return SpeakerDiarizer(device=device)


# Public API — lazy, only if diarization dependencies are available.
speaker_diarizer: Optional[SpeakerDiarizer] = None
if _DIARIZATION_AVAILABLE:
    speaker_diarizer = create_diarizer()