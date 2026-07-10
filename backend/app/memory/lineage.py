"""Belief-chain lineage — reading the supersede history of a fact.

A corrected fact is never overwritten: a NEW row is written and the old one is retired
(is_superseded=True, superseded_by_id -> the successor). So a belief forms a linear chain
old -> ... -> live, where each retired row points FORWARD to what replaced it. Finding a row's
PREDECESSOR is therefore a reverse lookup on that pointer: "who has superseded_by_id == me?".

Two read shapes on that same backward walk:
- `_belief_chain_ids`: ONE fact, the FULL chain, one query per hop — for the history detail view.
- `predecessors_for`: MANY live facts, the IMMEDIATE predecessor of each, in two batched queries
  — for the recall hot path, where a per-fact loop would be an N+1.

Linearity invariant (load-bearing, enforced by the WRITE path, not the schema): each Memory is
superseded at most once, because `_search_similar` (pipeline.py) excludes is_superseded rows as
UPDATE targets so a retired row is never re-targeted, and every UPDATE mints a fresh successor.
`superseded_by_id` is indexed but NOT unique (migration 0009) — do not rely on the DB to hold this.
"""

import uuid
from dataclasses import dataclass

from sqlmodel import Session, col, select

from app.models import Memory


def _belief_chain_ids(session: Session, memory: Memory) -> list[uuid.UUID]:
    """Every memory row in this belief's lineage: the given row plus each predecessor that was
    superseded (transitively) into it. Because a correction writes a NEW row and retires the old
    one, the UPDATE history lands on the predecessor; walking the chain lets the live successor
    show its full ADD -> UPDATE -> ... lineage instead of a bare 'ADD'. Cycle-guarded."""
    ids = [memory.id]
    seen = {memory.id}
    cursor = memory.id
    while True:
        predecessor = session.exec(
            select(Memory).where(Memory.superseded_by_id == cursor)
        ).first()
        if predecessor is None or predecessor.id in seen:
            break
        ids.append(predecessor.id)
        seen.add(predecessor.id)
        cursor = predecessor.id
    return ids


@dataclass
class PredecessorInfo:
    """The immediate prior belief a live fact replaced, plus whether older revisions exist.

    Deliberately does NOT carry a timestamp: the moment a belief changed is the SUCCESSOR's
    created_at (the live fact), not this predecessor's birth — the caller sources that from the
    live row so it is never mislabeled.
    """

    content: str  # the retired belief's text — what the user "previously" believed
    has_older: bool  # True if that predecessor itself superseded an even older belief


def predecessors_for(
    session: Session, live_fact_ids: list[uuid.UUID]
) -> dict[uuid.UUID, PredecessorInfo]:
    """For MANY live facts at once, the IMMEDIATE predecessor of each — in TWO column-projected,
    batched queries (never N+1, never the 1536-dim embedding). Keyed by the LIVE fact's id; a fact
    with no predecessor is simply absent from the returned dict."""
    if not live_fact_ids:
        return {}
    # query 1: each predecessor row (a retired row whose successor is one of our live facts).
    # PROJECTED columns only — loading full ORM rows would drag the Vector(1536) embedding per row.
    rows = session.exec(
        select(Memory.id, Memory.content, Memory.superseded_by_id).where(
            col(Memory.superseded_by_id).in_(live_fact_ids)
        )
    ).all()
    if not rows:
        return {}
    predecessor_ids = [pred_id for pred_id, _content, _successor_id in rows]
    # query 2: which of those predecessors THEMSELVES superseded an older belief — i.e. some row
    # points at them via superseded_by_id. The returned values ARE those predecessor ids.
    older = set(
        session.exec(
            select(Memory.superseded_by_id).where(
                col(Memory.superseded_by_id).in_(predecessor_ids)
            )
        ).all()
    )
    return {
        successor_id: PredecessorInfo(content=content, has_older=pred_id in older)
        for pred_id, content, successor_id in rows
    }
