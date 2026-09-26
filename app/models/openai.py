from __future__ import annotations

import time
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from app.models.solver import OcrResult


class ImageUrl(BaseModel):
    url: str  # data:image/png;base64,... or a plain http(s) URL


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Union[TextContentPart, ImageContentPart]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, list[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Two-phase OCR support: results previously returned by POST /v1/ocr,
    # echoed back verbatim so a client can OCR 15-20 screenshots across
    # several small requests (each capped by Vercel's 4.5 MB body limit)
    # and still solve them as ONE problem. Superseded numbering is
    # re-normalised server-side, so ordering just needs to be preserved.
    ocr_results: Optional[list[OcrResult]] = None
    # Whether multiple screenshots should be treated as consecutive views
    # of ONE continuously-scrolled problem (the default: repeated chrome
    # and scroll-seam overlap between them is detected and removed) or as
    # independent items that must never be merged/deduplicated (set this
    # to false when screenshots are of different problems, a problem plus
    # unrelated material, or otherwise not one continuous scroll — see
    # app/services/problem_reconstructor.py for why leaving this on for
    # unrelated screenshots can silently delete real content).
    stitch: bool = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class OpenAIErrorBody(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorBody


def make_error(message: str, error_type: str, code: str | None = None, param: str | None = None) -> dict:
    return OpenAIErrorResponse(error=OpenAIErrorBody(message=message, type=error_type, code=code, param=param)).model_dump()
