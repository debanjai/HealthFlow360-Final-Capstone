"""Private, in-memory PDF/DOCX radiology-report extraction (not RAG corpus indexing)."""
"""
06 Report Ingest: Ingest and extract text from radiology reports
Module: 06 Report Ingest
Author: Debanjali Ghosh
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REPORT_CHARS = 30_000


from .observability import traced, span, update

@traced("ingest_report_upload")
def ingest_report_upload(uploaded) -> str:
    """Extract report text from a Streamlit UploadedFile without persisting it."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {'.pdf', '.docx'}:
        raise ValueError('Upload a PDF or DOCX radiology report.')
    data = uploaded.getvalue()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Report must be non-empty and at most 10 MB.')
    if suffix == '.pdf':
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError('PDF ingestion requires pymupdf.') from exc
        try:
            with pymupdf.open(stream=data, filetype='pdf') as pdf:
                if pdf.is_encrypted:
                    raise ValueError('Password-protected PDFs are not supported.')
                if len(pdf) > 40:
                    raise ValueError('PDF exceeds the 40-page limit.')
                text = '\n'.join(page.get_text('text') for page in pdf)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('Could not read this PDF.') from exc
    else:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError('DOCX ingestion requires python-docx.') from exc
        try:
            document = Document(BytesIO(data))
            parts = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    parts.append(' | '.join(cell.text for cell in row.cells))
            text = '\n'.join(parts)
        except Exception as exc:
            raise ValueError('Could not read this DOCX.') from exc
    text = text.strip()
    if not text:
        raise ValueError('No extractable text found. Scanned/image-only PDFs require OCR and are not supported.')
    if len(text) > MAX_REPORT_CHARS:
        raise ValueError('Extracted report exceeds 30,000 characters.')
    return text
