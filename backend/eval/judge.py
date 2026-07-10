"""LLM-as-judge: grade a predicted answer against the gold answer.

Free-text answers can't be scored by string match ("7 May 2023" vs "May 7, 2023" vs "the 7th"
all mean the same). So a SEPARATE, stronger model reads the question, the gold answer, and our
answer, and rules whether they convey the same fact — 1 (correct) or 0 (wrong). Averaged over
all questions, that mean is the J score the whole field reports.

The judge MUST differ from the model that generated the answer: a model tends to favor its own
outputs (self-preference bias, Panickssery et al., NeurIPS 2024, arXiv:2404.13076), so judging
gpt-4o-mini with gpt-4o-mini would flatter the score. Default judge = Claude via the same
gateway; set EVAL_JUDGE_MODEL to override (openai/gpt-4o is a safe fallback if the gateway does
not expose the Anthropic slug — the run smoke-tests the judge model before spending on a full run).
"""

import json
import os
from dataclasses import dataclass


def _parse_first_json(text: str) -> dict:
    """Read the FIRST JSON object in the reply, ignoring any prose or a second object after it.

    The judge (Claude) occasionally returns the verdict object followed by a trailing sentence or
    a second object; a plain json.loads raises 'Extra data' on that. raw_decode stops at the end of
    the first object, so it is robust to trailing content. Returns {} if nothing parses (→ a
    conservative 'wrong' verdict rather than a crash)."""
    start = text.find("{")
    if start == -1:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}

# A different, stronger model than the gpt-4o-mini generator (verified present on the gateway).
# Override with EVAL_JUDGE_MODEL; openai/gpt-4o is a safe alternate.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"


def judge_model() -> str:
    return os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


JUDGE_SYSTEM = """You grade a predicted answer against a gold answer for a question about a person.

Mark it CORRECT if the prediction conveys the same factual answer as the gold answer, allowing
for differences in wording, date format, or extra harmless detail. Mark it WRONG if it states a
different fact, omits the key information the gold answer gives, contradicts the gold answer, or
says it does not know when the gold answer is a real fact.

Respond with strict JSON: {"correct": true or false, "reason": "one short sentence"}."""


def build_judge_messages(question: str, gold: str, predicted: str) -> list[dict]:
    user = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct?"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


@dataclass
class Verdict:
    correct: bool
    reason: str


def judge(client, model: str, question: str, gold: str, predicted: str) -> Verdict:
    """One grading call. Deterministic (temperature 0) so a re-run reproduces the score."""
    response = client.chat.completions.create(
        model=model,
        messages=build_judge_messages(question, gold, predicted),
        temperature=0,
    )
    data = _parse_first_json(response.choices[0].message.content or "{}")
    return Verdict(correct=bool(data.get("correct")), reason=str(data.get("reason", "")))