"""The two LLM prompts of the Mem0 pipeline, plus the summary prompt.

Faithful to arXiv 2504.19413's two-phase design (extraction -> ADD/UPDATE/DELETE/NOOP),
with one deliberate divergence: NO deception. Mem0's own OSS extraction prompt instructs
the model to claim facts "came from publicly available sources on the internet"; we do the
opposite — every fact traces to its source conversation, and provenance is a product feature.
"""

import json
from collections.abc import Sequence

# ── Phase 1: extraction ──────────────────────────────────────────────────────
# Read the conversation, distil the durable facts, drop the chit-chat.

FACT_EXTRACTION_SYSTEM = """You extract durable, reusable facts about the USER.

CRITICAL SCOPE: extract facts ONLY from the messages under "New exchange". The summary and
recent messages are BACKGROUND CONTEXT ONLY — never extract or repeat a fact that appears only
there. If the new exchange introduces no new durable fact, return an empty list.

A durable fact is worth remembering across sessions: identity, preferences, plans,
relationships, possessions, professional details, important events. Write each as one short,
self-contained sentence in the third person (e.g. "Works as a backend developer").

Do NOT extract: greetings, thanks, small talk, the assistant's own words, or transient
questions ("what's the weather"). When in doubt, extract nothing.

Respond with strict JSON: {"facts": ["fact one", "fact two"]}"""


def build_extraction_messages(
    summary: str, recent: Sequence[str], exchange: Sequence[dict]
) -> list[dict]:
    """Prompt = (running conversation summary, recent message window, the new exchange)."""
    parts: list[str] = []
    if summary:
        parts.append(f"Conversation summary so far:\n{summary}")
    if recent:
        parts.append("Recent messages:\n" + "\n".join(recent))
    new_lines = "\n".join(f'{m["role"]}: {m["content"]}' for m in exchange)
    parts.append("New exchange (extract facts from this):\n" + new_lines)
    return [
        {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ── Phase 2: the ADD / UPDATE / DELETE / NOOP decision ───────────────────────
# One candidate fact vs. the most similar existing memories -> one operation.

MEMORY_DECISION_SYSTEM = """You maintain a user's memory. Given ONE new fact and a list of the
user's most similar existing memories, choose exactly one operation:

- ADD: the fact is new; nothing equivalent exists.
- UPDATE: an existing memory covers the same topic but the new fact corrects or enriches it.
  Return the richer, corrected text.
- DELETE: the new fact makes an existing memory false or obsolete.
- NOOP: the fact is already captured, or is not worth storing.

Respond with strict JSON:
{"event": "ADD|UPDATE|DELETE|NOOP", "target_index": <index of the existing memory for
UPDATE/DELETE, else null>, "text": "<final memory text for ADD/UPDATE, else empty>"}

Examples:
New fact "Works as a backend developer" with existing [0] "Works as a frontend developer"
-> {"event": "UPDATE", "target_index": 0, "text": "Works as a backend developer"}
New fact "Loves hiking" with existing [0] "Works as a backend developer"
-> {"event": "ADD", "target_index": null, "text": "Loves hiking"}
New fact "No longer uses Python" with existing [0] "Loves working with Python"
-> {"event": "DELETE", "target_index": 0, "text": ""}"""


def build_decision_messages(fact: str, similar: Sequence[str]) -> list[dict]:
    if similar:
        listing = "\n".join(f"[{i}] {m}" for i, m in enumerate(similar))
    else:
        listing = "(none)"
    user = f"Existing related memories:\n{listing}\n\nNew fact: {fact}"
    return [
        {"role": "system", "content": MEMORY_DECISION_SYSTEM},
        {"role": "user", "content": user},
    ]


# ── The chat surface (Phase 2): answer with remembered context ───────────────

CHAT_SYSTEM = """You are a warm, helpful assistant with long-term memory of this user.

Your memory below has two parts: remembered facts, and photo memories — descriptions of the
photos and screenshots the user fed into their memory, each with its capture date. You do not
see image pixels, but the photo memories ARE the content of the user's images. When the user
asks about their photos or something in them, answer directly from the photo memories and
mention when each was captured. NEVER say you "cannot access images" — the photo memories are
your access. Use remembered facts to personalise naturally. If neither part covers what is
asked, answer normally. NEVER invent or assume facts about the user that are not in the memory
or the conversation.

What you remember about this user:
{memories}

From the user's photos and screenshots (with capture dates):
{photos}"""


def build_chat_messages(
    memories: Sequence[str], photos: Sequence[str], history: Sequence[dict], message: str
) -> list[dict]:
    memory_block = "\n".join(f"- {m}" for m in memories) if memories else "(nothing yet)"
    photo_block = "\n".join(f"- {p}" for p in photos) if photos else "(no photos yet)"
    msgs: list[dict] = [
        {"role": "system", "content": CHAT_SYSTEM.format(memories=memory_block, photos=photo_block)}
    ]
    msgs.extend(history)  # prior turns of this conversation, for short-term continuity
    msgs.append({"role": "user", "content": message})
    return msgs


# ── The asynchronous summary refresher ───────────────────────────────────────

SUMMARY_SYSTEM = """Summarise the conversation below into a concise running summary (3-5
sentences) that captures who the user is and what has been discussed. Respond with strict
JSON: {"summary": "..."}"""


def build_summary_messages(episodes: Sequence[str]) -> list[dict]:
    body = "\n".join(episodes)
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": body},
    ]


# ── Style inference: read HOW the user writes off their own messages ─────────
# Nobody states their communication style; they demonstrate it. This prompt reads a sample of
# the user's own messages and distils the manner — never the content — into durable traits.

STYLE_INFERENCE_SYSTEM = """You infer a user's COMMUNICATION STYLE from messages they wrote.

Describe only HOW they write — tone, length, directness, formality, structure, emoji and
punctuation habits, language mix — never WHAT they write about. Do not extract topics, facts,
plans, or content preferences; only the manner of writing.

Return 2-4 short, durable trait sentences in the third person, each starting with
"Communication style:" (e.g. "Communication style: writes short, direct messages and prefers
answers that lead with the conclusion"). State only traits the sample clearly demonstrates —
if the messages support fewer traits, return fewer.

Respond with strict JSON: {"traits": ["trait one", "trait two"]}"""


def build_style_messages(samples: Sequence[str]) -> list[dict]:
    body = "\n".join(f"- {s}" for s in samples)
    return [
        {"role": "system", "content": STYLE_INFERENCE_SYSTEM},
        {"role": "user", "content": f"Messages the user wrote (oldest first):\n{body}"},
    ]


# ── Hybrid retrieval: decompose a question into structured filters ───────────

QUERY_DECOMPOSE_SYSTEM = """You turn a question to a personal memory into structured search
filters. Today is {today}. Extract ONLY what the question explicitly implies — do not invent
filters.

Respond with strict JSON, exactly this shape:
{{"entities": ["specific people/pets/things named, e.g. Monty"],
  "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
  "place": "a place name if the question names one, else null",
  "semantic_query": "a short declarative rewrite of the question for similarity search",
  "wants_all": true if it asks for a count or 'all'/'every', else false}}

Rules:
- A bare year ("2023") -> start 2023-01-01, end 2023-12-31. "last year" -> the previous calendar
  year relative to today. A month -> that month's first and last day. Resolve relative dates
  against today.
- If the question names no specific entity, time, or place, return empty entities, null
  time_range, null place, wants_all false — it is a general question and needs no filters."""


def build_decompose_messages(message: str, today: str) -> list[dict]:
    return [
        {"role": "system", "content": QUERY_DECOMPOSE_SYSTEM.format(today=today)},
        {"role": "user", "content": message},
    ]


def parse_json(content: str) -> dict:
    """Parse an LLM JSON reply, tolerating code fences or prose around the object."""
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced {...} object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object found in LLM reply: {content!r}")