from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_PATH = Path(__file__).parent / "data" / "nlu_kg_quality_benchmark.jsonl"

REQUIRED_TOP_LEVEL_KEYS = {
    "id",
    "scenario_tag",
    "acceptance_tags",
    "input",
    "prior_context",
    "expected_entities",
    "expected_relations",
    "expected_conflicts",
    "expected_intent",
}

ALLOWED_SCENARIO_TAGS = {
    "pronoun_heavy_dialogue",
    "alias_heavy_entities",
    "conflicting_facts",
    "temporal_updates",
    "ambiguous_intent",
    "long_context_continuation",
}

ALLOWED_ENTITY_TYPES = {"person", "location", "item", "creature", "event"}

ALLOWED_INTENTS = {
    "explore",
    "action",
    "dialogue",
    "use_item",
    "ask_info",
    "rest",
    "trade",
}


def _load_cases() -> list[dict]:
    raw_lines = BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
    lines = [line for line in raw_lines if line.strip()]
    return [json.loads(line) for line in lines]


def test_benchmark_file_exists_and_has_minimum_cases() -> None:
    assert BENCHMARK_PATH.exists(), f"Missing benchmark file: {BENCHMARK_PATH}"
    cases = _load_cases()
    assert len(cases) >= 120, f"Expected >=120 cases, got {len(cases)}"


def test_required_keys_and_non_empty_values() -> None:
    for idx, case in enumerate(_load_cases()):
        missing = REQUIRED_TOP_LEVEL_KEYS - set(case.keys())
        assert not missing, f"Case[{idx}] missing keys: {sorted(missing)}"

        assert isinstance(case["id"], str) and case["id"].strip(), f"Case[{idx}] invalid id"
        assert isinstance(case["scenario_tag"], str) and case["scenario_tag"].strip(), (
            f"Case[{idx}] invalid scenario_tag"
        )
        assert isinstance(case["acceptance_tags"], list) and case["acceptance_tags"], (
            f"Case[{idx}] acceptance_tags must be non-empty list"
        )
        assert isinstance(case["input"], str) and case["input"].strip(), (
            f"Case[{idx}] input must be non-empty string"
        )
        assert isinstance(case["prior_context"], list) and case["prior_context"], (
            f"Case[{idx}] prior_context must be non-empty list"
        )
        assert isinstance(case["expected_entities"], list), (
            f"Case[{idx}] expected_entities must be list"
        )
        assert isinstance(case["expected_relations"], list), (
            f"Case[{idx}] expected_relations must be list"
        )
        assert isinstance(case["expected_conflicts"], list), (
            f"Case[{idx}] expected_conflicts must be list"
        )
        assert case["expected_intent"] in ALLOWED_INTENTS, (
            f"Case[{idx}] invalid expected_intent: {case['expected_intent']}"
        )


def test_allowed_tags_and_unique_ids() -> None:
    cases = _load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "Benchmark ids must be unique"

    seen_tags = set()
    for idx, case in enumerate(cases):
        tag = case["scenario_tag"]
        assert tag in ALLOWED_SCENARIO_TAGS, f"Case[{idx}] unknown scenario_tag: {tag}"
        seen_tags.add(tag)

    assert seen_tags == ALLOWED_SCENARIO_TAGS, (
        f"Scenario coverage mismatch. seen={sorted(seen_tags)}"
    )


def test_entity_and_relation_shape() -> None:
    for idx, case in enumerate(_load_cases()):
        for e_idx, entity in enumerate(case["expected_entities"]):
            assert isinstance(entity, dict), f"Case[{idx}] entity[{e_idx}] must be object"
            assert set(entity.keys()) >= {"name", "type"}, (
                f"Case[{idx}] entity[{e_idx}] missing required keys"
            )
            assert isinstance(entity["name"], str) and entity["name"].strip(), (
                f"Case[{idx}] entity[{e_idx}] invalid name"
            )
            assert entity["type"] in ALLOWED_ENTITY_TYPES, (
                f"Case[{idx}] entity[{e_idx}] invalid type: {entity['type']}"
            )

        for r_idx, relation in enumerate(case["expected_relations"]):
            assert isinstance(relation, dict), f"Case[{idx}] relation[{r_idx}] must be object"
            assert set(relation.keys()) >= {"source", "target", "relation"}, (
                f"Case[{idx}] relation[{r_idx}] missing required keys"
            )
            assert isinstance(relation["source"], str) and relation["source"].strip(), (
                f"Case[{idx}] relation[{r_idx}] invalid source"
            )
            assert isinstance(relation["target"], str) and relation["target"].strip(), (
                f"Case[{idx}] relation[{r_idx}] invalid target"
            )
            assert isinstance(relation["relation"], str) and relation["relation"].strip(), (
                f"Case[{idx}] relation[{r_idx}] invalid relation type"
            )


def test_conflict_shape() -> None:
    for idx, case in enumerate(_load_cases()):
        for c_idx, conflict in enumerate(case["expected_conflicts"]):
            assert isinstance(conflict, dict), f"Case[{idx}] conflict[{c_idx}] must be object"
            assert set(conflict.keys()) >= {"type", "entity"}, (
                f"Case[{idx}] conflict[{c_idx}] missing required keys"
            )
            assert isinstance(conflict["type"], str) and conflict["type"].strip(), (
                f"Case[{idx}] conflict[{c_idx}] invalid type"
            )
            assert isinstance(conflict["entity"], str) and conflict["entity"].strip(), (
                f"Case[{idx}] conflict[{c_idx}] invalid entity"
            )
