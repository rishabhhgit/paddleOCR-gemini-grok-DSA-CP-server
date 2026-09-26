from __future__ import annotations

from pydantic import BaseModel


class ExtractedImage(BaseModel):
    mime_type: str
    data: bytes  # raw decoded bytes, kept in memory only


class OcrResult(BaseModel):
    index: int  # original screenshot order
    text: str
    unreadable: bool = False


class ReconstructedProblem(BaseModel):
    text: str
    screenshot_count: int
