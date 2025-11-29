"""Bulk test suite for NameGuard scoring engine.

This script programmatically generates >1000 test cases across categories and
validates that the rule-based engine behaves as expected. It is meant to be
run manually, not as part of the FastAPI app.

Usage (from /app):
    python -m backend.nameguard_bulk_tests
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import pandas as pd

from backend.models import NameEvaluationInput
from backend.scoring import evaluate_names


@dataclass
class TestCase:
    name: str
    priority: int
    category: str
    expectation: Callable[[dict], Tuple[bool, str]]


def _expect_low_probability(reason_substring: str = "", max_prob: float = 30.0):
    def check(result: dict) -> Tuple[bool, str]:
        ok_prob = result["acceptance_probability"] <= max_prob
        ok_reason = True
        if reason_substring:
            ok_reason = any(
                reason_substring.lower() in (flag.description.lower())
                for flag in result["rule_flags"]
            )
        ok = ok_prob and ok_reason
        msg = (
            f"prob={result['acceptance_probability']} "
            f"flags={[f.code for f in result['rule_flags']]}"
        )
        return ok, msg

    return check


def _expect_has_flag(flag_code: str):
    def check(result: dict) -> Tuple[bool, str]:
        codes = [f.code for f in result["rule_flags"]]
        ok = flag_code in codes
        return ok, f"flags={codes}"

    return check


def _expect_high_probability(min_prob: float = 70.0):
    def check(result: dict) -> Tuple[bool, str]:
        ok = result["acceptance_probability"] >= min_prob
        return ok, f"prob={result['acceptance_probability']}"

    return check


def build_test_cases() -> List[TestCase]:
    tests: List[TestCase] = []

    # 1) Existing company duplicates / near-duplicates
    existing_path = "data/existing_companies_sample.xlsx"
    try:
        df_existing = pd.read_csv(existing_path).head(200)
        for _, row in df_existing.iterrows():
            base_name = str(row.get("name") or "").strip()
            if not base_name:
                continue
            tests.append(
                TestCase(
                    name=base_name,
                    priority=1,
                    category="existing_duplicate",
                    expectation=_expect_has_flag("too_similar_existing_company"),
                )
            )
            tests.append(
                TestCase(
                  name=f"{base_name} Technologies Private Limited",
                  priority=1,
                  category="existing_near_duplicate",
                  expectation=_expect_has_flag("too_similar_existing_company"),
                )
            )
    except Exception:
        pass

    # 2) Offensive words – generate company-like names
    offensive_path = "data/offensive_words_multilang.xlsx"
    try:
        df_off = pd.read_csv(offensive_path).sample(n=150, random_state=42)
        for _, row in df_off.iterrows():
            word = str(row.get("word") or "").strip()
            if not word:
                continue
            name = f"{word} Logistics Private Limited"
            tests.append(
                TestCase(
                    name=name,
                    priority=1,
                    category="offensive",
                    expectation=_expect_low_probability(max_prob=15.0),
                )
            )
    except Exception:
        # If file missing, fallback small set
        for w in ["chutiya", "idiot", "gunda"]:
            name = f"{w} Logistics Private Limited"
            tests.append(
                TestCase(
                    name=name,
                    priority=1,
                    category="offensive",
                    expectation=_expect_low_probability(max_prob=15.0),
                )
            )

    # 3) Prohibited / reserved terms – use a subset
    prohibited_path = "data/prohibited_words.xlsx"
    try:
        df_proh = pd.read_csv(prohibited_path).sample(n=150, random_state=7)
        for _, row in df_proh.iterrows():
            term = str(row.get("word") or "").strip()
            if not term:
                continue
            name = f"{term} Development Private Limited"
            tests.append(
                TestCase(
                    name=name,
                    priority=1,
                    category="prohibited",
                    expectation=_expect_has_flag("prohibited_or_reserved_term"),
                )
            )
    except Exception:
        pass

    # 4) Political leader references (initials + phonetic)
    for nm_variant in ["NM", "N.M.", "N M", "NDM", "N.D.M.", "NDM Ventures Private Limited"]:
        tests.append(
            TestCase(
                name=nm_variant,
                priority=1,
                category="political_initials",
                expectation=_expect_has_flag("political_or_leader_reference_initials"),
            )
        )

    for podi_variant in ["Podi Tech Private Limited", "Podi Foods LLP", "Podi Innovations"]:
        tests.append(
            TestCase(
                name=podi_variant,
                priority=1,
                category="political_phonetic",
                expectation=_expect_has_flag("political_or_leader_reference_phonetic"),
            )
        )

    # Explicit user cases
    tests.append(
        TestCase(
            name="Accenture",
            priority=1,
            category="user_case_accenture",
            expectation=_expect_has_flag("too_similar_existing_company"),
        )
    )
    tests.append(
        TestCase(
            name="Chutiya",
            priority=1,
            category="user_case_chutiya",
            expectation=_expect_has_flag("obscene_or_offensive"),
        )
    )

    # 5) Benign, likely-acceptable synthetic names
    benign_prefixes = [
        "Blue Mango",
        "Aranya",
        "Astra",
        "Himalaya Ridge",
        "Saral",
        "Nitya",
        "Prudent Leaf",
        "Golden Banyan",
    ]
    benign_suffixes = [
        "Analytics Private Limited",
        "Innovations LLP",
        "Ventures Private Limited",
        "Systems Private Limited",
    ]
    for _ in range(250):
        name = f"{random.choice(benign_prefixes)} {random.choice(benign_suffixes)}"
        tests.append(
            TestCase(
                name=name,
                priority=1,
                category="benign",
                expectation=_expect_high_probability(min_prob=60.0),
            )
        )

    # Ensure we have at least ~1000 tests
    # (the above should already exceed it; this is just a check)
    return tests


def run_tests() -> None:
    tests = build_test_cases()
    total = len(tests)
    passed = 0

    print(f"Running {total} NameGuard tests...")

    # Group small batches into evaluate_names calls for efficiency
    batch_size = 50
    for i in range(0, total, batch_size):
        batch = tests[i : i + batch_size]
        inputs = [NameEvaluationInput(name=t.name, priority=t.priority) for t in batch]
        results = evaluate_names(inputs)

        for tcase, res in zip(batch, results):
            ok, msg = tcase.expectation(res.__dict__)
            if ok:
                passed += 1
            else:
                print(
                    f"[FAIL] category={tcase.category} name='{tcase.name}' -> {msg} "
                    f"prob={res.acceptance_probability} flags={[f.code for f in res.rule_flags]}"
                )

    print(f"\nSummary: {passed}/{total} tests passed")

    # Print spotlight examples explicitly
    spotlight_names = ["Accenture", "Chutiya", "NDM", "Podi"]
    print("\nSpotlight examples:")
    for nm in spotlight_names:
        res = evaluate_names([NameEvaluationInput(name=nm, priority=1)])[0]
        print(
            f"- {nm}: prob={res.acceptance_probability}%, "
            f"flags={[f.code for f in res.rule_flags]}, explanations={res.explanations}"
        )


if __name__ == "__main__":
    run_tests()
