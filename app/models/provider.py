from __future__ import annotations

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    name: str
    model: str
    endpoint: str
    api_key: str
    status: str = "active"
