'''Decides which cache entry a request is allowed to match
Idea: a request is split into two parts.
  1. query_text -> the last user message. This gets embedded and compared by MEANING.
  2. scope_hash -> everything else (model, temperature, max_tokens, system prompt, earlier chat).
     This must match EXACTLY. If it differs, the cached answer is never reused.
This way, only the last message is compared semantically.
'''
import hashlib
import json
from dataclasses import dataclass

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def content_to_text(content) -> str:
    """Message content is usually a string, but OpenAI also allows a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        else:
            parts.append(json.dumps(part, sort_keys=True))
    return "\n".join(parts)


@dataclass(frozen=True)
class CacheRequest:
    model: str
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None

    def last_user_index(self) -> int:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                return i
        raise ValueError("Request has no user message")

    @property
    def query_text(self) -> str:
        """The text we turn into an embedding: the last user message."""
        return content_to_text(self.messages[self.last_user_index()].get("content")).strip()

    @property
    def system_prompt_hash(self) -> str:
        """Hash of all system/developer messages."""
        system_text = "\n".join(
            content_to_text(m.get("content"))
            for m in self.messages
            if m.get("role") in ("system", "developer")
        )
        return sha(system_text)

    @property
    def scope_hash(self) -> str:
        """Hash of everything that must match exactly."""
        idx = self.last_user_index()
        other_messages = self.messages[:idx] + self.messages[idx + 1:]
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "other_messages": other_messages,
        }
        return sha(json.dumps(payload, sort_keys=True, default=str))