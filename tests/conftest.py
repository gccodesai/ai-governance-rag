import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import load_chunks                      # noqa: E402
from rag.llm import LLMResponse                  # noqa: E402
from rag.retriever import Retriever              # noqa: E402


class FakeLLM:
    """Stands in for Gemini/Groq: returns scripted replies and records the prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else "OK [1]"
        return LLMResponse(text=reply, model="fake-model", input_tokens=100, output_tokens=20)


@pytest.fixture(scope="session")
def retriever():
    return Retriever(load_chunks(), use_dense=False)
