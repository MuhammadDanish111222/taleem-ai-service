"""Safe, typed retrieval DTOs and the deliberately conservative evidence policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceStrength(StrEnum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class RetrievalChannel(StrEnum):
    DENSE = "dense"
    EXPECTED_QUESTION = "expected_question"
    LEXICAL = "lexical"


@dataclass(frozen=True)
class RetrievalScope:
    board_id: str
    class_id: str
    subject_id: str
    chapter_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            value.strip() for value in (self.board_id, self.class_id, self.subject_id)
        ):
            raise ValueError("RETRIEVAL_SCOPE_FIELDS_REQUIRED")
        if self.chapter_id is not None and not self.chapter_id.strip():
            raise ValueError("RETRIEVAL_CHAPTER_ID_BLANK")


@dataclass(frozen=True)
class RetrievedVisual:
    visual_id: str
    title: str
    description: str
    display_policy: str


@dataclass(frozen=True)
class Citation:
    """A safe source record for later answer generation; it contains no vectors or storage IDs."""

    citation_id: str
    content: str
    chapter_id: str | None
    topic_no: str | None
    topic_title: str | None
    page_start: int | None
    page_end: int | None
    visuals: tuple[RetrievedVisual, ...] = ()


@dataclass(frozen=True)
class ChannelContribution:
    channel: RetrievalChannel
    rank: int


@dataclass(frozen=True)
class RetrievedEvidence:
    citation: Citation
    fused_rank: int
    contributions: tuple[ChannelContribution, ...]


@dataclass(frozen=True)
class EvidenceResult:
    strength: EvidenceStrength
    results: tuple[RetrievedEvidence, ...]
    reason: str


def classify_evidence(results: tuple[RetrievedEvidence, ...]) -> EvidenceResult:
    """Apply the approved rank/channel evidence rule to the top fused parent only.

    Retrieval values remain ranks rather than confidence or probability values.
    A contribution at rank 1--3 can qualify for strong evidence only alongside
    a distinct qualifying channel on the same top parent chunk. This is kept
    out of runtime settings because relaxing it would weaken grounded-answer
    integrity rather than tune an operational resource limit.
    """
    if not results:
        return EvidenceResult(
            strength=EvidenceStrength.NONE,
            results=(),
            reason="NO_SCOPED_EVIDENCE",
        )
    qualifying_channels = {
        contribution.channel
        for contribution in results[0].contributions
        if contribution.rank <= 3
    }
    if len(qualifying_channels) >= 2:
        return EvidenceResult(
            strength=EvidenceStrength.STRONG,
            results=results,
            reason="TOP_PARENT_MULTI_CHANNEL_TOP_THREE",
        )
    return EvidenceResult(
        strength=EvidenceStrength.WEAK,
        results=results,
        reason="TOP_PARENT_DOES_NOT_MEET_MULTI_CHANNEL_TOP_THREE",
    )
