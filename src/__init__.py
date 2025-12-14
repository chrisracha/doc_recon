"""
Document Reconstruction Pipeline
================================

A modular OCR and document reconstruction system for academic documents.
Converts scanned PDFs/images into structured JSON, Markdown, DOCX, and PDF.

Main components:
- Image preprocessing (deskew, denoise, binarize)
- Layout detection (titles, paragraphs, equations, tables, figures)
- Text OCR with confidence scoring
- Math/equation OCR to LaTeX
- Table extraction and reconstruction
- Multi-format export
"""

__version__ = "1.0.0"
__author__ = "Document Reconstruction Team"


