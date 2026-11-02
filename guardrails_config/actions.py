"""NeMo Guardrails custom action: asks the local guard classifier (FastAPI) whether a user message is unsafe."""
import os

import httpx
from nemoguardrails.actions import action

GUARD_URL = os.environ.get("GUARD_URL", "http://localhost:8100")


@action(is_system_action=True)
async def guard_is_unsafe(text: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{GUARD_URL}/classify", json={"text": text})
        r.raise_for_status()
    return r.json()["label"] == "unsafe"
