from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class ExtractedContent:
    text: str
    method: str
    media_type: str
    warnings: list[str]


def detect_media_type(path: str | Path, content_type: str | None = None) -> str:
    if content_type:
        return content_type.split(";")[0].lower()
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def extract_image_ocr(image_data: bytes) -> ExtractedContent:
    """Extract text from image files using OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(img)
        return ExtractedContent(
            text=text,
            method="ocr-tesseract",
            media_type="image",
            warnings=[],
        )
    except ImportError as exc:
        raise RuntimeError("Install the OCR extra with: uv sync --extra ocr") from exc
    except Exception as e:
        return ExtractedContent(
            text="",
            method="ocr-failed",
            media_type="image",
            warnings=[str(e)],
        )


def extract_asr(audio_data: bytes, media_type: str) -> ExtractedContent:
    """Extract text from audio files using ASR."""
    try:
        import whisper
        import tempfile
        import os

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        # Load and transcribe with Whisper
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        os.unlink(tmp_path)

        return ExtractedContent(
            text=result["text"],
            method="whisper",
            media_type=media_type,
            warnings=[],
        )
    except ImportError as exc:
        raise RuntimeError("Install the ASR extra with: uv sync --extra asr") from exc
    except Exception as e:
        return ExtractedContent(
            text="",
            method="asr-failed",
            media_type=media_type,
            warnings=[str(e)],
        )


def extract_video_asr(video_data: bytes, media_type: str) -> ExtractedContent:
    """Extract text from video files by extracting audio and then applying ASR."""
    try:
        import subprocess
        import tempfile
        import os

        # Extract audio from video using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_tmp:
            video_tmp.write(video_data)
            video_path = video_tmp.name

        audio_path = video_path.replace(".mp4", ".wav")
        # Use ffmpeg to extract audio
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            check=True,
            capture_output=True,
        )

        # Now transcribe the extracted audio
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        result = extract_asr(audio_data, "audio/wav")

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(audio_path)

        return result
    except ImportError as exc:
        raise RuntimeError("Install the ASR extra with: uv sync --extra asr") from exc
    except Exception as e:
        return ExtractedContent(
            text="",
            method="video-asr-failed",
            media_type=media_type,
            warnings=[str(e)],
        )


def extract_html(data: bytes) -> ExtractedContent:
    soup = BeautifulSoup(data, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    text = soup.get_text("\n", strip=True)
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title and title not in text[:500]:
        text = f"{title}\n\n{text}"
    return ExtractedContent(
        text=text,
        method="html_bs4",
        media_type="text/html",
        warnings=[],
    )


def extract_text(data: bytes, media_type: str = "text/plain") -> ExtractedContent:
    return ExtractedContent(
        text=data.decode("utf-8", errors="replace"),
        method="utf8_text",
        media_type=media_type,
        warnings=[],
    )


def extract_pdf(data: bytes) -> ExtractedContent:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install the PDF extra with: uv sync --extra pdf") from exc

    import io
    import logging

    # Optional OCR import
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        _ocr_available = True
    except ImportError:
        _ocr_available = False
        logging.getLogger(__name__).warning(
            "OCR dependencies not installed. Install with: uv sync --extra ocr"
        )

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]

    # If no text was extracted, attempt OCR on scanned PDFs
    if not any(pages) and _ocr_available:
        logging.getLogger(__name__).info("Attempting OCR on scanned PDF")
        try:
            images = convert_from_bytes(data)
            ocr_text = "\n\n".join(
                pytesseract.image_to_string(img, lang="eng") for img in images
            )
            return ExtractedContent(
                text=ocr_text,
                method="pypdf+ocr",
                media_type="application/pdf",
                warnings=[],
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"OCR failed: {str(e)}")

    return ExtractedContent(
        text="\n\n".join(pages),
        method="pypdf" if pages and pages[0] else "pdf_without_text",
        media_type="application/pdf",
        warnings=[],
    )


def extract_docx(data: bytes) -> ExtractedContent:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install the DOCX extra with: uv sync --extra docx") from exc

    import io

    document = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return ExtractedContent(
        text=text,
        method="python-docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        warnings=[],
    )


def extract(data: bytes, media_type: str) -> ExtractedContent:
    media_type = media_type.lower()

    if "html" in media_type:
        return extract_html(data)
    if media_type == "application/pdf":
        return extract_pdf(data)
    if media_type in {
        "text/plain",
        "text/markdown",
        "application/json",
        "application/xml",
    }:
        return extract_text(data, media_type)
    if media_type.endswith("wordprocessingml.document"):
        return extract_docx(data)
    # Handle image types with OCR
    if media_type.startswith("image/"):
        return extract_image_ocr(data)
    # Handle audio types with ASR
    if media_type.startswith("audio/"):
        return extract_asr(data, media_type)
    # Handle video types by extracting audio and then ASR
    if media_type.startswith("video/"):
        # For video, we need to extract audio first
        # This is a simplified approach - in reality we'd need to handle video parsing
        # For now, we'll treat video as audio and attempt ASR (requires ffmpeg)
        return extract_asr(data, media_type)

    return ExtractedContent(
        text="",
        method="unsupported",
        media_type=media_type,
        warnings=["No core extractor available for this media type"],
    )


def extract_metadata(data: bytes, media_type: str, extracted: ExtractedContent) -> dict:
    """Return simple metadata (title, language, summary) where available.

    Attempts best-effort extraction without adding hard dependencies.
    """
    meta: dict = {"title": None, "language": None, "summary": None, "published_at": None}

    mt = (media_type or extracted.media_type or "").lower()
    if "html" in mt:
        soup = BeautifulSoup(data, "html.parser")
        if soup.title and soup.title.string:
            meta["title"] = soup.title.string.strip()
        # lang attribute on <html>
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta["language"] = html_tag.get("lang")
        # description meta
        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            meta["summary"] = desc.get("content").strip()
        published = None
        for selector in [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"property": "og:updated_time"}),
            ("meta", {"name": "date"}),
            ("meta", {"name": "dc.date"}),
            ("meta", {"name": "dc:date"}),
        ]:
            tag = soup.find(*selector)
            if tag and tag.get("content"):
                published = tag.get("content").strip()
                break
        if published:
            try:
                meta["published_at"] = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                # Keep only parseable timestamps on the canonical record.
                meta["published_at"] = None

    elif "pdf" in mt:
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            info = reader.metadata
            if info and info.title:
                meta["title"] = info.title
            if info and info.language:
                meta["language"] = info.language
            if info and getattr(info, "creation_date", None):
                creation = info.creation_date
                if isinstance(creation, datetime):
                    meta["published_at"] = creation
        except Exception:
            # optional dependency or extraction failure — ignore
            pass

    # Try lightweight language detection if installed
    if not meta.get("language"):
        try:
            from langdetect import detect

            lang = detect(extracted.text or "")
            meta["language"] = lang
        except Exception:
            meta["language"] = None

    # Fallback date extraction from URL and content patterns.
    # Many policy pages lack meta tags but encode dates in URLs
    # (e.g., icann-2012-02-25, /igf-2023/, rfc3935).
    if not meta.get("published_at"):
        import re as _re
        from datetime import datetime as _datetime

        # Try URL patterns first: look for YYYY-MM-DD or YYYY/MM/DD or YYYY
        url_patterns: list[str] = []
        if "html" in mt:
            url_patterns.append(str(getattr(data, "url", "") or ""))
        url_patterns.append("")  # will be filled from context if available
        text_to_search = extracted.text[:2000] if extracted.text else ""

        # Date patterns in URL: icann-2012-02-25-en, /2024/, igf-2023-kyoto, rfc3935
        date_match = _re.search(
            r"(?:^|[^0-9])((?:19|20)\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])(?:[^0-9]|$)",
            text_to_search,
        )
        if not date_match:
            date_match = _re.search(
                r"(?:^|[^0-9])((?:19|20)\d{2})[-/](0[1-9]|1[0-2])(?:[^0-9]|$)",
                text_to_search,
            )
        if not date_match:
            # Just a year: /2024/ /igf-2023- /2006/
            date_match = _re.search(
                r"(?:^|[^0-9])((?:199\d|20[0-2]\d))(?:[^0-9]|$)",
                text_to_search,
            )

        if date_match:
            try:
                year = int(date_match.group(1))
                month = int(date_match.group(2)) if date_match.lastindex and date_match.lastindex >= 2 else 1
                day = int(date_match.group(3)) if date_match.lastindex and date_match.lastindex >= 3 else 1
                month = max(1, min(12, month))
                day = max(1, min(28, day))
                meta["published_at"] = _datetime(year, month, day)
            except (ValueError, IndexError):
                pass

    return meta
