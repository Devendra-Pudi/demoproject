"""Curated, pre-validated demo answers for model-free previews.

Retrieval still runs for real in demo mode. A curated quote is only used when retrieval returns
the chunk that contains it *verbatim*, and the selection is re-checked with the same citation
invariant that guards live model output (:func:`ai_portfolio.rag.enforce_citations`). If no curated
case matches the question, or a quote no longer matches the shipped documents, no demo answer is
produced and the service returns a clearly labeled retrieval-only response instead.

This module exists so the portfolio has a clickable, honest demo path with no model download.
It is not a language model and must never be described as one.
"""
from __future__ import annotations

import json
import re

from .rag import enforce_citations
from .retrieval import Chunk

# Questions are matched by token overlap; quotes must be exact substrings of the shipped documents.
DEMO_CASES: list[dict] = [
    {"question": "When must production API keys be rotated?",
     "source": "security.md", "quote": "Production API keys must be rotated every 90 days."},
    {"question": "How many days of paid annual leave do full-time employees receive?",
     "source": "leave.md", "quote": "Full-time employees receive 20 days of paid annual leave."},
    {"question": "How many days per week may employees work remotely?",
     "source": "leave.md",
     "quote": "Employees may work remotely up to three days per week with manager approval."},
    {"question": "When must lost devices be reported?",
     "source": "security.md",
     "quote": "Lost devices must be reported to the security team within 24 hours."},
    {"question": "When must employees submit expense reports?",
     "source": "expenses.md",
     "quote": "Employees must submit expense reports within 30 days of purchase."},
    {"question": "What is the daily meal reimbursement limit?",
     "source": "expenses.md",
     "quote": "The daily meal reimbursement limit is 60 dollars during business travel."},
    {"question": "What is the strongest second factor for company accounts?",
     "source": "security.md", "quote": "Hardware security keys are the preferred second factor."},
    {"question": "How long are financial records retained?",
     "source": "data-retention.md", "quote": "Financial records are retained for seven years."},
    {"question": "How quickly must a deletion request from a data subject be completed?",
     "source": "data-retention.md",
     "quote": "Deletion requests received from data subjects must be completed within thirty days"},
    {"question": "What is the annual learning budget per employee?",
     "source": "learning-and-development.md",
     "quote": "Each employee has an annual learning budget of 1,500 dollars."},
    {"question": "How quickly must suspected exposure of customer data be reported?",
     "source": "customer-data-handling.md",
     "quote": "Suspected exposure of customer data must be reported to the security team within one hour of discovery"},
    {"question": "What is the minimum password length?",
     "source": "password-and-access.md",
     "quote": "Passwords must be at least sixteen characters and unique to each service."},
    {"question": "How far in advance must flights be booked?",
     "source": "travel.md",
     "quote": "Flights must be booked at least fourteen days in advance through the approved travel tool."},
    {"question": "What is the hotel booking cap in standard cities?",
     "source": "travel.md",
     "quote": "Hotel bookings are capped at 220 dollars per night in standard cities"},
    {"question": "How quickly must a non-compliant device be remediated?",
     "source": "workplace-devices.md",
     "quote": "Non-compliance detected by the device management tool must be resolved within five business days"},
    {"question": "May employees paste confidential customer data into an AI assistant?",
     "source": "ai-usage-policy.md",
     "quote": "Approved tools must never be given confidential customer data, credentials or unreleased financial results."},
    {"question": "What is the limit on gifts from suppliers?",
     "source": "code-of-conduct.md", "quote": "Gifts worth more than 100 dollars must be declined or reported."},
    {"question": "How are security incidents reported?",
     "source": "incident-response.md",
     "quote": "Security incidents must be reported immediately through the on-call channel."},
    {"question": "When must a severity one incident be reviewed?",
     "source": "incident-response.md",
     "quote": "Severity one incidents require a written timeline within forty-eight hours and a blameless review within ten business days."},
    {"question": "What approval is needed for purchases above 5,000 dollars?",
     "source": "procurement.md",
     "quote": "Purchases above 5,000 dollars require two competing quotes and director approval."},
    {"question": "How often do formal performance reviews take place?",
     "source": "performance-reviews.md",
     "quote": "Formal performance reviews take place twice a year, in June and December."},
    {"question": "When is the office closed over the winter holidays?",
     "source": "holiday-calendar.md",
     "quote": "Offices close between 24 December and 1 January."},
    {"question": "When must security awareness training be completed?",
     "source": "onboarding.md",
     "quote": "Security awareness training must be completed within the first week."},
    {"question": "What happens to company equipment when an employee leaves?",
     "source": "remote-work-equipment.md",
     "quote": "Equipment purchased by the company remains company property and must be returned within fourteen days of leaving."},
    {"question": "How must confidential data be stored?",
     "source": "data-classification.md",
     "quote": "Confidential and restricted data must be encrypted in transit and at rest."},
    {"question": "How are accidents and near misses reported?",
     "source": "health-and-safety.md",
     "quote": "Employees must report accidents and near misses to their manager on the same day."},
    {"question": "How is overtime compensated?",
     "source": "overtime-and-timesheets.md",
     "quote": "Overtime is compensated at 1.5 times the standard rate, or with time off in lieu where local law permits."},
    {"question": "How often are critical vendors reviewed?",
     "source": "vendor-risk.md", "quote": "Critical vendors are reviewed annually."},
]

# Below this token-overlap (Jaccard) score a question is treated as unmatched, so the demo
# abstains rather than answering a question nobody curated.
MATCH_THRESHOLD = 0.6


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def best_case(question: str) -> dict | None:
    wanted = _tokens(question)
    if not wanted:
        return None
    scored: list[tuple[float, dict]] = []
    for case in DEMO_CASES:
        case_tokens = _tokens(case["question"])
        union = wanted | case_tokens
        scored.append((len(wanted & case_tokens) / len(union) if union else 0.0, case))
    scored.sort(key=lambda item: -item[0])
    top_score, top_case = scored[0]
    if len(scored) > 1 and scored[1][0] == top_score:
        return None  # ambiguous: two curated questions match equally well
    return top_case if top_score >= MATCH_THRESHOLD else None


def demo_answer(question: str, chunks: list[Chunk]) -> list[dict] | None:
    """Return validated citations built from a curated case, or None when nothing matches."""
    case = best_case(question)
    if case is None:
        return None
    match = next((c for c in chunks
                  if c.source == case["source"] and case["quote"] in c.text), None)
    if match is None:
        return None  # the curated quote is no longer present in retrieved evidence
    try:
        return enforce_citations(
            json.dumps({"citations": [{"chunk_id": match.id, "quote": case["quote"]}]}), chunks)
    except ValueError:
        return None
