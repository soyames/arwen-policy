from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
    import numpy as np
    from PIL import Image

    _VIDEO_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    _VIDEO_AVAILABLE = False


class FFmpegVideoProcessor:
    """FFmpeg-based video processing with frame extraction and OCR overlay."""

    def __init__(
        self,
        output_dir: str = "frames",
        fps: float = 1.0,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        """Initialize video processor.

        Args:
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract (1 = one frame per second)
            frame_width: Width to resize frames
            frame_height: Height to resize frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

    def extract_frames(self, video_path: str, start_time: float = 0, duration: Optional[float] = None) -> List[str]:
        """Extract frames from video at specified FPS.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration to extract in seconds (None = entire video)

        Returns:
            List of extracted frame file paths
        """
        output_pattern = str(self.output_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-ss", str(start_time),
        ]

        if duration:
            cmd.extend(["-t", str(duration)])

        cmd.extend([
            "-i", video_path,
            "-vf", f"fps={self.fps},scale={self.frame_width}:{self.frame_height}:force_original_aspect_ratio=decrease,pad={self.frame_width}:{self.frame_height}:(ow-iw)/2:(oh-ih)/2",
            "-vcodec", "mjpeg",
            "-q:v", "2",  # High quality
            output_pattern
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr}")

        # Get list of extracted frames
        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        return [str(f) for f in frames]

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio track from video.

        Args:
            video_path: Path to video file
            output_path: Output audio path (default: same name with .wav)

        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = str(self.output_dir / f"{Path(video_path).stem}.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

        return output_path

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        info = json.loads(result.stdout)

        # Extract relevant info
        video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})

        return {
            "duration": float(info.get("format", {}).get("duration", 0)),
            "width": video_stream.get("width", 0),
            "height": video_stream.get("height", 0),
            "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
            "video_codec": video_stream.get("codec_name", "unknown"),
            "audio_codec": audio_stream.get("codec_name", "unknown"),
            "audio_channels": audio_stream.get("channels", 0),
            "audio_sample_rate": audio_stream.get("sample_rate", 0),
        }

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS string like '30000/1001' to float."""
        try:
            num, denom = fps_str.split("/")
            return float(num) / float(denom)
        except:
            return 0.0

    def run_ocr_on_frames(self, frame_paths: List[str], language: str = "eng") -> List[Dict[str, Any]]:
        """Run OCR on extracted frames.

        Args:
            frame_paths: List of frame image paths
            language: Tesseract language code

        Returns:
            List of OCR results with frame index and text
        """
        from arwen_etl.ocr import ocr_processor

        results = []
        for i, frame_path in enumerate(frame_paths):
            try:
                if ocr_processor is None:
                    raise RuntimeError("OCR processor unavailable (pytesseract not installed)")
                ocr_result = ocr_processor.process_image(frame_path)
                results.append({
                    "frame_index": i,
                    "frame_path": frame_path,
                    "text": ocr_result["text"],
                    "confidence": ocr_result["confidence"],
                    "timestamp": i / self.fps,  # Approximate timestamp
                })
            except Exception as e:
                results.append({
                    "frame_index": i,
                    "frame_path": frame_path,
                    "text": "",
                    "confidence": 0.0,
                    "timestamp": i / self.fps,
                    "error": str(e),
                })
        return results

    def process_video(
        self,
        video_path: str,
        asr_processor: Optional[Any] = None,
        language: str = "eng",
        extract_audio: bool = True,
    ) -> Dict[str, Any]:
        """Full video processing pipeline.

        Args:
            video_path: Path to video file
            asr_processor: Optional ASR processor (WhisperASRProcessor)
            language: Language code for OCR/ASR
            extract_audio: Whether to extract and transcribe audio

        Returns:
            Dictionary with video metadata, frames, OCR results, and transcript
        """
        # Get video info
        video_info = self.get_video_info(video_path)

        # Extract frames
        frame_paths = self.extract_frames(video_path)

        # Run OCR on frames
        ocr_results = self.run_ocr_on_frames(frame_paths, language)

        # Extract and transcribe audio if requested
        transcript = None
        diarization = None
        if extract_audio and asr_processor is not None:
            try:
                audio_path = self.extract_audio(video_path)
                asr_result = asr_processor.transcribe(audio_path, language=language[:2] if language else None)
                transcript = asr_result

                # If diarization is available, run it
                if hasattr(asr_processor, 'diarize'):
                    diarization = asr_processor.diarize(audio_path)
            except Exception as e:
                transcript = {"error": str(e)}

        return {
            "video_path": video_path,
            "video_info": video_info,
            "frame_count": len(frame_paths),
            "frames": ocr_results,
            "transcript": transcript,
            "diarization": diarization,
            "language": language,
        }

    def overlay_ocr_text(self, frame_path: str, text: str, output_path: Optional[str] = None) -> str:
        """Overlay OCR text on video frame for visualization.

        Args:
            frame_path: Path to frame image
            text: Text to overlay
            output_path: Output path (default: annotated_ prefix)

        Returns:
            Path to annotated frame
        """
        if output_path is None:
            output_path = str(Path(frame_path).parent / f"annotated_{Path(frame_path).name}")

        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # Truncate text if too long
        max_chars = 80
        display_text = text[:max_chars] + ("..." if len(text) > max_chars else "")

        # Draw background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        x = 10
        y = 10
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + 5),
            (0, 128, 0),  # Green background
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            frame,
            display_text,
            (x, y),
            font,
            font_scale,
            (255, 255, 0),  # Yellow text
            thickness,
            cv2.LINE_AA
        )

        cv2.imwrite(output_path, frame)
        return output_path


# Factory function
def create_video_processor(
    output_dir: str = "frames",
    fps: float = 1.0,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> FFmpegVideoProcessor:
    """Create video processor with sensible defaults."""
    if not _VIDEO_AVAILABLE:
        raise RuntimeError(
            "Video processing dependencies not installed. "
            "Install with: uv sync --extra ocr"
        )
    return FFmpegVideoProcessor(
        output_dir=output_dir,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )


# Public API — lazy, only if video deps are available.
video_processor: Optional[FFmpegVideoProcessor] = None
if _VIDEO_AVAILABLE:
    video_processor = create_video_processor()