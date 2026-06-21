"""Critique panel: orchestrates critic agents and renders a consolidated report."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .base import CriticAgent, CritiqueContext, Finding, Severity


@dataclass
class CritiqueReport:
    findings: List[Finding] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    def counts_by_severity(self) -> Dict[Severity, int]:
        counter = Counter(f.severity for f in self.findings)
        return {sev: counter.get(sev, 0) for sev in sorted(Severity, reverse=True)}

    def by_aspect(self) -> Dict[str, List[Finding]]:
        grouped: Dict[str, List[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.aspect, []).append(finding)
        for items in grouped.values():
            items.sort(key=lambda f: f.severity, reverse=True)
        return grouped

    def to_markdown(self) -> str:
        lines = ["# PRIMER/DIPS algorithm critique", ""]
        counts = self.counts_by_severity()
        total = len(self.findings)
        summary = ", ".join("%s: %d" % (sev.label(), counts[sev]) for sev in counts)
        lines.append("**%d findings** (%s)" % (total, summary))
        lines.append("")

        for aspect, items in self.by_aspect().items():
            lines.append("## %s" % aspect)
            lines.append("")
            for finding in items:
                lines.append(finding.to_markdown())
                lines.append("")

        if self.errors:
            lines.append("## Agent errors")
            lines.append("")
            for name, err in self.errors.items():
                lines.append("- **%s**: %s" % (name, err))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_text(self) -> str:
        counts = self.counts_by_severity()
        summary = "  ".join("%s=%d" % (sev.label(), counts[sev]) for sev in counts)
        lines = ["PRIMER/DIPS algorithm critique", "=" * 64,
                 "%d findings   %s" % (len(self.findings), summary), ""]
        for aspect, items in self.by_aspect().items():
            lines.append("[%s]" % aspect)
            for finding in items:
                lines.append("  (%-8s) %s" % (finding.severity.label(), finding.title))
                if finding.evidence:
                    lines.append("            evidence: %s" % finding.evidence)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class CritiquePanel:
    """Runs a set of critic agents and aggregates their findings."""

    def __init__(self, agents: Sequence[CriticAgent]):
        self.agents = list(agents)

    def run(self, context: CritiqueContext) -> CritiqueReport:
        report = CritiqueReport()
        for agent in self.agents:
            try:
                report.findings.extend(agent.review(context))
            except Exception as exc:  # one agent must not break the panel
                report.errors[agent.name] = "%s: %s" % (type(exc).__name__, exc)
        report.findings.sort(key=lambda f: f.severity, reverse=True)
        return report


def default_panel() -> CritiquePanel:
    from .agents import ALL_AGENTS

    return CritiquePanel([agent_cls() for agent_cls in ALL_AGENTS])
