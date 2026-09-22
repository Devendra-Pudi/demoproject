# 01 · Ask My Policies — production-pattern RAG

**Code:** `src/ai_portfolio/retrieval.py`, `rag.py`, `api.py`; fixtures in `data/docs` and `data/eval`.

## Retrieval design

1. Split Markdown paragraphs into at most 180 words with 30-word overlap for long paragraphs. Stable source/content SHA-256 IDs identify evidence.
2. BM25 (`k1=1.5`, `b=0.75`) finds exact terms. MiniLM normalized embeddings support semantic search using cosine/dot similarity.
3. Reciprocal-rank fusion (`k=60`) merges sparse and dense top-20 candidates without treating their raw scores as comparable.
4. The MS MARCO MiniLM cross-encoder reranks query/passage pairs; keep top four.
5. Ollama selects structured `{chunk_id, quote}` citations. The server rejects unknown IDs, changed text, malformed JSON and additional fields.
6. Only validated **verbatim quotes** become answer text. Empty selections abstain. Failures show retrieved passages as a clearly labeled non-answer.

This intentionally sacrifices paraphrasing to enforce a strong provenance invariant. Source documents remain untrusted data. Exact quote checks prevent fabricated provenance but do not solve malicious source content, relevance, access control, or all prompt injection.

## Run and evaluate

Follow the root full-neural setup, then:

```bash
ai-eval --mode neural --generate --output artifacts/neural-generation.json
# For later same-environment regression comparison:
ai-eval --mode neural --generate --baseline artifacts/neural-generation.json \
  --output artifacts/neural-generation-next.json
```

Evaluation checks evidence-bearing chunk Recall@4 and MRR, not just document hits. Unanswerable examples test abstention only when `--generate` is enabled. Exact evidence match is a conservative deterministic proxy, not an LLM judge. Generation failures fail the command; they are not converted to passing scores.

## Gates

Recall@4 ≥ .95; MRR ≥ .80; citation invariants = 1.0; retrieval p95 ≤ 3 seconds. With generation: evidence match ≥ .80; abstention accuracy = 1.0. Baseline recall/MRR drops > .02 fail; p95 > 1.5× baseline fails with a 50 ms noise floor. Smoke and neural baselines cannot be mixed.

CI's smoke gate checks mechanics with test doubles. Dispatch the neural workflow and run generation evaluation before release. Use a representative, versioned held-out domain set (including adversarial, irrelevant, conflicting and stale-document cases) rather than relying on the eight included examples.
