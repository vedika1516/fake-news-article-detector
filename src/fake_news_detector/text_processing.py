from __future__ import annotations

import re
from typing import Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - fallback for minimal environments
    PorterStemmer = None


STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{2,}\b")


class SimpleFallbackStemmer:
    """Small fallback stemmer used when nltk is unavailable."""

    def stem(self, token: str) -> str:
        for suffix in ("ingly", "edly", "ing", "edly", "edly", "ed", "ly", "ies", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                if suffix == "ies":
                    return token[:-3] + "y"
                return token[: -len(suffix)]
        return token


STEMMER = PorterStemmer() if PorterStemmer is not None else SimpleFallbackStemmer()


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    if not token or token in STOPWORDS:
        return ""
    return STEMMER.stem(token)


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = ""
    tokens = [normalize_token(match.group()) for match in TOKEN_PATTERN.finditer(text.lower())]
    return [token for token in tokens if token]


def preprocess_text(text: str) -> str:
    return " ".join(tokenize(text))


def merge_text_fields(title: str, text: str) -> str:
    title = title if isinstance(title, str) else ""
    text = text if isinstance(text, str) else ""
    return f"{title.strip()} {text.strip()}".strip()


def preprocess_corpus(texts: Iterable[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]
