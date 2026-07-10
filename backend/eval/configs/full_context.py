"""The 'full_context' baseline — the quality ceiling and the cost floor.

No memory, no retrieval: the ENTIRE conversation is stuffed into the prompt for every question.
It usually wins on quality (it can see everything) and loses badly on cost and latency (tens of
thousands of tokens per question). Reproducing that trade-off — a quality ceiling at a punishing
price — is the honest backdrop that makes our engine's economics meaningful, and it's exactly
the comparison the paper leads with.

There is no ingest step: the whole conversation IS the context, handed in fresh per question.
"""

from app.config import Settings
from eval.common import AnswerResult, timed_answer


def answer(
    client, settings: Settings, question: str, *, dialogue: list[str]
) -> AnswerResult:
    transcript = "\n".join(dialogue)
    # "retrieval" is trivial (the whole transcript, every time) but still routed through
    # timed_answer so latency + token accounting is identical to the other two configs.
    return timed_answer(lambda: transcript, client, settings.llm_model, question)