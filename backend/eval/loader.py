"""Load the LOCOMO benchmark into typed samples.

LOCOMO (github.com/snap-research/locomo, CC BY-NC — study-only) is 10 long, multi-session
conversations between two speakers, each annotated with ~200 human-written question/gold-answer
pairs. It is the standard yardstick memory systems (Mem0, Zep, ...) report on, so a number
measured here is comparable to theirs. This module downloads the dataset once (gitignored) and
parses its real on-disk schema into frozen dataclasses the eval configs consume.

Two schema facts that bite, both verified against the real file:
  - a gold `answer` is sometimes a bare int (e.g. a year `2022`), so it is coerced to str.
  - category 5 (adversarial) carries NO `answer` — it has `adversarial_answer` and tests refusal,
    so it is excluded from the answerable set (a later round scores it separately).
"""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_PATH = Path(__file__).parent / "data" / "locomo10.json"

# LOCOMO's integer category labels, verified against the dataset (distribution across all 10:
# multi_hop 282, temporal 321, open_domain 96, single_hop 841, adversarial 446).
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
# category 5 tests "know when you can't answer" — no gold answer, scored differently — later round
ANSWERABLE = (1, 2, 3, 4)


@dataclass(frozen=True)
class Turn:
    speaker: str  # the real speaker name, e.g. "Caroline" (both speakers, not user/assistant)
    dia_id: str   # dialogue id, e.g. "D1:3" — what a qa's `evidence` points at
    text: str


@dataclass(frozen=True)
class DialogueSession:
    index: int         # 1-based session number
    date_time: str     # human string, e.g. "1:56 pm on 8 May, 2023" — the temporal anchor
    turns: list[Turn]


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str          # the gold answer, always a string (coerced from int where needed)
    category: int
    category_name: str
    evidence: list[str]  # dia_ids that contain the answer (used later for retrieval recall)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[DialogueSession]
    qa: list[QAPair]

    def dialogue_lines(self) -> list[str]:
        """The whole conversation as flat, chronological, speaker-attributed lines.

        Used verbatim by the full-context config (dump everything) and chunked by naive RAG.
        A date header precedes each session so temporal questions have something to bind to.
        """
        lines: list[str] = []
        for s in self.sessions:
            lines.append(f"[{s.date_time}]")
            lines.extend(f"{t.speaker}: {t.text}" for t in s.turns)
        return lines


def download_if_missing() -> Path:
    """Fetch the 2.8 MB dataset once into the gitignored data dir. No-op if already present."""
    if DATA_PATH.exists():
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def _parse_sessions(conversation: dict) -> list[DialogueSession]:
    """Pull the `session_<n>` turn-lists out of the flat conversation dict, in order.

    The conversation dict mixes real sessions (`session_1`) with sibling metadata
    (`session_1_date_time`, `session_summary`, `observation`) — we keep only the numbered
    turn-lists and pair each with its `_date_time`.
    """
    indices = sorted(
        int(key.split("_")[1])
        for key in conversation
        if key.startswith("session_")
        and key.split("_")[1].isdigit()  # excludes session_summary / observation / event_summary
        and not key.endswith("date_time")
    )
    sessions: list[DialogueSession] = []
    for i in indices:
        turns: list[Turn] = []
        for t in conversation[f"session_{i}"]:
            # a pure-image turn has no `text` but a `blip_caption` — keep its textual trace
            text = t.get("text") or (
                f"[shared a photo: {t['blip_caption']}]" if t.get("blip_caption") else ""
            )
            if not text:
                continue
            turns.append(Turn(speaker=t["speaker"], dia_id=t.get("dia_id", ""), text=text))
        sessions.append(
            DialogueSession(
                index=i,
                date_time=conversation.get(f"session_{i}_date_time", ""),
                turns=turns,
            )
        )
    return sessions


def _parse_qa(raw_qa: list[dict], categories: tuple[int, ...]) -> list[QAPair]:
    pairs: list[QAPair] = []
    for q in raw_qa:
        category = q.get("category")
        if category not in categories or "answer" not in q:  # skip adversarial (no gold answer)
            continue
        pairs.append(
            QAPair(
                question=q["question"],
                answer=str(q["answer"]).strip(),  # gold can be an int (a year) — coerce to str
                category=category,
                category_name=CATEGORY_NAMES.get(category, str(category)),
                evidence=list(q.get("evidence") or []),
            )
        )
    return pairs


def load_samples(
    limit: int | None = None, categories: tuple[int, ...] = ANSWERABLE
) -> list[Sample]:
    """Parse the LOCOMO file into samples.

    limit caps how many of the 10 conversations to load (2 for a cheap first run); categories
    filters the questions (default = the 4 answerable ones).
    """
    raw = json.loads(download_if_missing().read_text(encoding="utf-8"))
    samples: list[Sample] = []
    for s in raw[:limit] if limit else raw:
        conversation = s["conversation"]
        samples.append(
            Sample(
                sample_id=s["sample_id"],
                speaker_a=conversation["speaker_a"],
                speaker_b=conversation["speaker_b"],
                sessions=_parse_sessions(conversation),
                qa=_parse_qa(s.get("qa", []), categories),
            )
        )
    return samples