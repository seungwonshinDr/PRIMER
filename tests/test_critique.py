from primer2.critique import (CritiqueContext, Severity, default_panel)
from primer2.critique.agents import ALL_AGENTS
from primer2.critique.base import CriticAgent, Finding


def _run():
    context = CritiqueContext.default(seed=0)
    return default_panel().run(context)


def test_every_agent_is_a_critic_and_runs():
    context = CritiqueContext.default(seed=0)
    for agent_cls in ALL_AGENTS:
        agent = agent_cls()
        assert isinstance(agent, CriticAgent)
        findings = agent.review(context)
        assert findings, "%s produced no findings" % agent.name
        assert all(isinstance(f, Finding) for f in findings)


def test_panel_aggregates_without_errors():
    report = _run()
    assert not report.errors, report.errors
    assert len(report.findings) >= 20


def test_findings_sorted_by_severity_desc():
    report = _run()
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, reverse=True)


def test_has_critical_findings_from_multiple_aspects():
    report = _run()
    critical_aspects = {f.aspect for f in report.findings
                        if f.severity == Severity.CRITICAL}
    assert len(critical_aspects) >= 3


def test_some_findings_carry_live_evidence():
    report = _run()
    assert sum(1 for f in report.findings if f.evidence) >= 4


def test_reports_render():
    report = _run()
    assert "critique" in report.to_text().lower()
    assert report.to_markdown().startswith("# PRIMER/DIPS")
