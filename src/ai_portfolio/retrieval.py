"""BM25 + dense retrieval, reciprocal-rank fusion, and cross-encoder reranking."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


def tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def load_chunks(directory: Path, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    if not 0 <= overlap < max_words:
        raise ValueError("Require 0 <= overlap < max_words")
    chunks = []
    for path in sorted(directory.rglob("*.md")):
        source = path.relative_to(directory).as_posix()
        for paragraph in path.read_text().split("\n\n"):
            words = paragraph.split()
            for start in range(0, len(words), max_words - overlap):
                text = " ".join(words[start:start + max_words])
                digest = hashlib.sha256(f"{source}:{text}".encode()).hexdigest()[:16]
                chunks.append(Chunk(digest, source, text))
                if start + max_words >= len(words):
                    break
    return list({c.id: c for c in chunks}.values())


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class SentenceEncoder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        return self.model.predict([(query, c.text) for c in chunks]).tolist()


class SmokeEncoder:
    """Deterministic lexical hash vectors, NOT semantic embeddings. CI/smoke only."""
    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 256), dtype=float)
        for row, text in enumerate(texts):
            for word in tokens(text):
                index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 256
                result[row, index] += 1
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-9)


class SmokeReranker:
    """Lexical overlap; deliberately not presented as a cross-encoder."""
    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(c.text))) / max(len(q), 1) for c in chunks]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], encoder: Encoder, reranker: Reranker):
        if not chunks:
            raise ValueError("No documents found")
        self.chunks, self.encoder, self.reranker = chunks, encoder, reranker
        self.counts = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg_length = max(float(np.mean(self.lengths)), 1)
        self.df = Counter(word for count in self.counts for word in count)
        self.vectors = encoder.encode([c.text for c in chunks])

    def bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for word in set(tokens(query)):
            df = self.df[word]
            idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            tf = np.array([c[word] for c in self.counts])
            scores += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length))
        return scores

    def search(self, query: str, k: int = 4, candidates: int = 20) -> list[Chunk]:
        if not query.strip() or k < 1 or candidates < k:
            raise ValueError("Nonempty query and candidates >= k >= 1 required")
        sparse = self.bm25(query)
        dense = self.vectors @ self.encoder.encode([query])[0]
        fused: dict[int, float] = {}
        for scores in (sparse, dense):
            for rank, idx in enumerate(np.argsort(-scores, kind="stable")[:candidates], 1):
                fused[int(idx)] = fused.get(int(idx), 0) + 1 / (60 + rank)
        selected = sorted(fused, key=lambda i: (-fused[i], i))[:candidates]
        pool = [self.chunks[i] for i in selected]
        scores = self.reranker.score(query, pool)
        return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-scores[i], i))[:k]]


def build_index(directory: Path, mode: str = "neural") -> HybridIndex:
    if mode not in {"smoke", "neural"}:
        raise ValueError("RETRIEVAL_MODE must be neural or smoke")
    encoder = SmokeEncoder() if mode == "smoke" else SentenceEncoder()
    reranker = SmokeReranker() if mode == "smoke" else CrossEncoderReranker()
    return HybridIndex(load_chunks(directory), encoder, reranker)
