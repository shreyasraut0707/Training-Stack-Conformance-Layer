"""The result object every probe returns (spec Section 8).

JSON, Markdown and JUnit reports (M1) are all built from this one object.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INFO = "info"
    NOT_TESTED = "not_tested"


class EvidenceClass(str, Enum):
    SEEN = "seen"          # observed directly; may drive a fail
    INFERRED = "inferred"  # follows from seen facts by a stated rule; may not drive a fail alone
    UNKNOWN = "unknown"    # could not be determined; never a fail, always shown


@dataclass
class Finding:
    evidence_class: EvidenceClass
    claim: str
    evidence: dict[str, Any] = field(default_factory=dict)
    rule: str | None = None  # required for inferred findings

    def __post_init__(self) -> None:
        self.evidence_class = EvidenceClass(self.evidence_class)
        if self.evidence_class is EvidenceClass.INFERRED and not self.rule:
            raise ValueError("an inferred finding must state the rule it follows from")


@dataclass
class ProbeResult:
    probe: str  # e.g. "P2-loss-bearing-tokens"
    status: Status
    oracle: str
    environment: dict[str, Any]
    reproducer: str
    findings: list[Finding] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    noise_floor: float | None = None
    declared_intent: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    reason: str | None = None  # why the probe is not_tested

    def __post_init__(self) -> None:
        self.status = Status(self.status)
        if self.status is Status.FAIL and not any(
            f.evidence_class is EvidenceClass.SEEN for f in self.findings
        ):
            raise ValueError("a fail must be backed by at least one 'seen' finding")
        if self.status is Status.NOT_TESTED and not self.reason:
            raise ValueError("a not_tested result must say why")

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            self,
            dict_factory=lambda items: {
                k: (v.value if isinstance(v, Enum) else v) for k, v in items
            },
        )


def summarize(results: list[ProbeResult]) -> str:
    """'4 pass, 1 fail, 2 not tested' -- deliberately no overall pass label."""
    counts = Counter(r.status for r in results)
    parts = [f"{counts[s]} {s.value.replace('_', ' ')}" for s in Status if counts[s]]
    return ", ".join(parts) or "no probes run"


def exit_code(results: list[ProbeResult], strict: bool = False) -> int:
    """0 if nothing failed, 1 if any probe failed (--strict also counts not_tested).

    Exit code 2 (usage or internal error) is raised by the CLI, not here.
    """
    statuses = {r.status for r in results}
    if Status.FAIL in statuses or (strict and Status.NOT_TESTED in statuses):
        return 1
    return 0
