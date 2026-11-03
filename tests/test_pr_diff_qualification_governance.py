"""Execute governed PR-differential qualification tests in existing Tier 1 CI.

This file only runs tests. It does not switch RQ-02 qualification policy, create
approval manifests, or authorize a production release.
"""
from pathlib import Path
import importlib.util
import unittest

import pytest

pytestmark = pytest.mark.tier1

RELEASE = Path(__file__).resolve().parents[1] / "scripts" / "release"


def load_suite(filename):
    location = RELEASE / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    assert suite.countTestCases() > 0, f"No tests discovered in {filename}"
    return suite


def summarize(result):
    if result.wasSuccessful():
        return
    entries = [
        f"{case.id()}: {detail}"
        for case, detail in (result.failures + result.errors)
    ]
    raise AssertionError(
        f"Governance adversarial tests failed: "
        f"ran={result.testsRun}, failures={len(result.failures)}, "
        f"errors={len(result.errors)}\n" + "\n".join(entries)
    )


def test_pr_diff_mocked_negative_controls():
    suite = load_suite("test_pr_diff_qualification.py")
    result = unittest.TestResult()
    suite.run(result)
    assert result.testsRun >= 12, f"Unexpected mock test census: {result.testsRun}"
    summarize(result)


def test_pr_diff_real_git_negative_controls():
    suite = load_suite("test_pr_diff_git_integration.py")
    result = unittest.TestResult()
    suite.run(result)
    assert result.testsRun >= 11, f"Unexpected Git integration census: {result.testsRun}"
    summarize(result)
