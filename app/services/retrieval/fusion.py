"""Deterministic rank-only reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.services.retrieval.evidence import (
    ChannelContribution,
    Citation,
    RetrievalChannel,
    RetrievedEvidence,
)

RRF_K = 60


@dataclass(frozen=True)
class RankedChannelHit:
    citation: Citation
    channel: RetrievalChannel
    rank: int


def fuse_ranked_hits(
    hits: Iterable[RankedChannelHit], rrf_k: int = RRF_K
) -> tuple[RetrievedEvidence, ...]:
    """Fuse channel ranks only; raw distances and lexical scores never cross this boundary."""
    if rrf_k < 1:
        raise ValueError("RRF_K_MUST_BE_POSITIVE")
    grouped: dict[str, list[RankedChannelHit]] = defaultdict(list)
    for hit in hits:
        if hit.rank < 1:
            raise ValueError("CHANNEL_RANK_MUST_BE_POSITIVE")
        grouped[hit.citation.citation_id].append(hit)

    candidates = []
    for citation_id, candidate_hits in grouped.items():
        # One hit per channel is expected. Keeping the best protects this boundary
        # if a future repository implementation returns duplicates.
        by_channel: dict[RetrievalChannel, RankedChannelHit] = {}
        for hit in candidate_hits:
            current = by_channel.get(hit.channel)
            if current is None or hit.rank < current.rank:
                by_channel[hit.channel] = hit
        contributions = tuple(
            ChannelContribution(
                channel=channel,
                rank=hit.rank,
            )
            for channel, hit in sorted(by_channel.items(), key=lambda item: item[0].value)
        )
        candidates.append(
            (
                citation_id,
                by_channel,
                contributions,
                sum(1.0 / (rrf_k + item.rank) for item in contributions),
            )
        )

    candidates.sort(key=lambda item: (-item[3], item[0]))
    return tuple(
        RetrievedEvidence(
            citation=next(iter(by_channel.values())).citation,
            fused_rank=index,
            contributions=contributions,
        )
        for index, (_, by_channel, contributions, weight) in enumerate(candidates, start=1)
    )
