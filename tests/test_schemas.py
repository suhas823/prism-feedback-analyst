"""Tests for LLM output schemas and citation verification."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insights.cluster_analyst import verify_citations
from src.insights.schemas import ClusterAnalysis


VALID = {
    "theme_name": "Playback stops in background",
    "summary": "Users report music halting when the app is backgrounded.",
    "root_causes": ["Aggressive OS battery optimization may kill the service"],
    "severity": 4,
    "severity_rationale": "Interrupts the core listening experience.",
    "recommended_actions": [
        {
            "action": "Detect battery-optimization kills and guide users to whitelist",
            "effort": "medium",
            "rationale": "Addresses the most common trigger without OS changes",
        }
    ],
    "evidence_quote_ids": ["play_store_review:1", "support_ticket:2"],
}


class TestClusterAnalysisSchema:
    def test_valid_payload_parses(self):
        a = ClusterAnalysis.model_validate(VALID)
        assert a.severity == 4

    def test_severity_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ClusterAnalysis.model_validate({**VALID, "severity": 6})

    def test_bad_effort_rejected(self):
        bad = {
            **VALID,
            "recommended_actions": [
                {"action": "x", "effort": "gigantic", "rationale": "y"}
            ],
        }
        with pytest.raises(ValidationError):
            ClusterAnalysis.model_validate(bad)

    def test_empty_actions_rejected(self):
        with pytest.raises(ValidationError):
            ClusterAnalysis.model_validate({**VALID, "recommended_actions": []})

    def test_parses_from_json_string(self):
        import json

        a = ClusterAnalysis.model_validate_json(json.dumps(VALID))
        assert a.theme_name.startswith("Playback")


class TestCitationVerification:
    members = {"play_store_review:abc-1", "support_ticket:2", "play_store_review:3"}

    def test_all_citations_valid(self):
        a = ClusterAnalysis.model_validate(
            {**VALID, "evidence_quote_ids": ["play_store_review:abc-1", "support_ticket:2"]}
        )
        ok, detail, resolved = verify_citations(a, self.members)
        assert ok
        assert "verified" in detail
        assert resolved == ["play_store_review:abc-1", "support_ticket:2"]

    def test_unknown_citation_fails(self):
        a = ClusterAnalysis.model_validate(
            {**VALID, "evidence_quote_ids": ["play_store_review:abc-1", "made_up:99"]}
        )
        ok, detail, _ = verify_citations(a, self.members)
        assert not ok
        assert "unknown" in detail

    def test_no_citations_fails(self):
        a = ClusterAnalysis.model_validate({**VALID, "evidence_quote_ids": []})
        ok, _, resolved = verify_citations(a, self.members)
        assert not ok
        assert resolved == []

    def test_prefix_dropped_id_resolves(self):
        # LLMs often cite the bare native id without the "source:" prefix.
        a = ClusterAnalysis.model_validate({**VALID, "evidence_quote_ids": ["abc-1"]})
        ok, _, resolved = verify_citations(a, self.members)
        assert ok
        assert resolved == ["play_store_review:abc-1"]

    def test_ambiguous_suffix_rejected(self):
        members = {"play_store_review:dup", "support_ticket:dup"}
        a = ClusterAnalysis.model_validate({**VALID, "evidence_quote_ids": ["dup"]})
        ok, detail, _ = verify_citations(a, members)
        assert not ok
        assert "unknown" in detail
