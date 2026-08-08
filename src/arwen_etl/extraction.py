from __future__ import annotations

import mimetypes
from dataclasses import dataclass
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

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return ExtractedContent(
        text="\n\n".join(pages),
        method="pypdf",
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
    meta: dict = {"title": None, "language": None, "summary": None}

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

    return meta
