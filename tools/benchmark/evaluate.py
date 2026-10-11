#!/usr/bin/env python3
"""Evaluate model-independent OCR predictions against a versioned golden set."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EVALUATOR_VERSION = "1.0.0"
REVIEW_REASONS = {
    "low_confidence",
    "score_exceeds_maximum",
    "unknown_question",
    "unresolved_student_identity",
    "ambiguous_outcome_mapping",
    "ambiguous_corrected_score",
    "inconsistent_page_identity",
    "source_region_mismatch",
    "conflicting_question_score",
    "invalid_candidate",
    "total_mismatch",
    "blank_field",
    "roster_near_match",
}
SCORE_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?\Z")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


class BenchmarkInputError(ValueError):
    """Raised when a benchmark input violates the versioned contract."""


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkInputError(f"{path}: expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkInputError(f"{path}: expected an array")
    return value


def _required(obj: dict[str, Any], keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise BenchmarkInputError(f"{path}: missing required field(s): {', '.join(missing)}")


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BenchmarkInputError(f"{path}: expected a non-empty string")
    if value != value.strip():
        raise BenchmarkInputError(f"{path}: leading/trailing whitespace is not canonical")
    return value


def _question_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _normalize_question_id(value):
        raise BenchmarkInputError(f"{path}: expected a non-empty question id")
    return value


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_QUESTION_PREFIXES = ("question", "soru", "q", "s")


def _normalize_question_id(value: str) -> str:
    """Mirror of the Rust `QuestionId::new` policy: `(S1)`, `S.1.` and `Soru 1` become `1`."""
    core = value.strip().translate(_ASCII_LOWER)
    while True:
        stripped = core.strip().lstrip("([").rstrip(")].:")
        if stripped == core:
            break
        core = stripped
    for prefix in _QUESTION_PREFIXES:
        if core.startswith(prefix):
            rest = core[len(prefix):].lstrip(" .:")
            if rest[:1] in tuple("0123456789"):
                core = rest
                break
    return core


def _nullable_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkInputError(f"{path}: expected an integer >= {minimum}")
    return value


def _decimal_score(value: Any, path: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str) or not SCORE_RE.fullmatch(value):
        raise BenchmarkInputError(
            f"{path}: expected a non-negative decimal string with at most two decimals"
        )
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise BenchmarkInputError(f"{path}: invalid decimal score") from exc


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkInputError(f"{path}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkInputError(f"{path}: expected a finite number")
    return result


def _validate_manifest(value: Any) -> None:
    manifest = _object(value, "prediction.manifest")
    fields = (
        "model_name",
        "model_revision",
        "artifact_sha256",
        "artifact_bytes",
        "runtime_name",
        "runtime_version",
        "backend",
        "thread_count",
        "temperature",
        "seed",
        "prompt_version",
        "timestamp",
        "evaluator_version",
    )
    _required(manifest, fields, "prediction.manifest")
    for key in (
        "model_name",
        "model_revision",
        "runtime_name",
        "runtime_version",
        "backend",
        "prompt_version",
        "evaluator_version",
    ):
        _text(manifest[key], f"prediction.manifest.{key}")
    timestamp = _text(manifest["timestamp"], "prediction.manifest.timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BenchmarkInputError("prediction.manifest.timestamp: expected ISO-8601 UTC") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timedelta(0):
        raise BenchmarkInputError("prediction.manifest.timestamp: expected an explicit UTC offset")
    if not isinstance(manifest["artifact_sha256"], str) or not SHA256_RE.fullmatch(
        manifest["artifact_sha256"]
    ):
        raise BenchmarkInputError("prediction.manifest.artifact_sha256: expected 64 hex characters")
    _integer(manifest["artifact_bytes"], "prediction.manifest.artifact_bytes", minimum=1)
    _integer(manifest["thread_count"], "prediction.manifest.thread_count", minimum=1)
    temperature = _finite_number(manifest["temperature"], "prediction.manifest.temperature")
    if temperature < 0:
        raise BenchmarkInputError("prediction.manifest.temperature: must be >= 0")
    seed = manifest["seed"]
    if seed != "not_applicable":
        _integer(seed, "prediction.manifest.seed")


def _valid_region(region: dict[str, Any], page_index: int) -> bool:
    if region["page_index"] != page_index:
        return False
    box = region["bbox"]
    x, y, width, height = (box[k] for k in ("x", "y", "width", "height"))
    return (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and width > 0.0
        and height > 0.0
        and x + width <= 1.0
        and y + height <= 1.0
    )


def _validate_region(value: Any, path: str, page_index: int, *, golden: bool) -> dict[str, Any]:
    region = _object(value, path)
    _required(region, ("field_key", "page_index", "bbox"), path)
    _text(region["field_key"], f"{path}.field_key")
    _integer(region["page_index"], f"{path}.page_index")
    box = _object(region["bbox"], f"{path}.bbox")
    _required(box, ("x", "y", "width", "height"), f"{path}.bbox")
    for key in ("x", "y", "width", "height"):
        box[key] = _finite_number(box[key], f"{path}.bbox.{key}")
    if golden and not _valid_region(region, page_index):
        raise BenchmarkInputError(f"{path}: golden regions must be valid normalized boxes on their page")
    return region


def _validate_page(value: Any, path: str, role: str) -> dict[str, Any]:
    page = _object(value, path)
    common = ("page_index", "student", "questions", "outcomes", "mappings", "regions")
    role_fields = ("review_required_fields",) if role == "golden" else (
        "parse_status",
        "parse_error",
        "review_issues",
        "confidence_by_field",
    )
    _required(page, common + role_fields, path)
    page_index = _integer(page["page_index"], f"{path}.page_index")

    student = _object(page["student"], f"{path}.student")
    _required(student, ("name", "number"), f"{path}.student")
    student["name"] = _nullable_text(student["name"], f"{path}.student.name")
    student["number"] = _nullable_text(student["number"], f"{path}.student.number")

    questions = _list(page["questions"], f"{path}.questions")
    question_ids: set[str] = set()
    for index, item in enumerate(questions):
        item_path = f"{path}.questions[{index}]"
        question = _object(item, item_path)
        _required(question, ("id", "earned_score", "maximum_score"), item_path)
        question["id"] = _question_id(question["id"], f"{item_path}.id")
        normalized_id = _normalize_question_id(question["id"])
        if normalized_id in question_ids:
            raise BenchmarkInputError(f"{item_path}.id: duplicate question id")
        question_ids.add(normalized_id)
        earned = _decimal_score(question["earned_score"], f"{item_path}.earned_score")
        maximum = _decimal_score(question["maximum_score"], f"{item_path}.maximum_score")
        if role == "golden" and earned is not None and maximum is not None and earned > maximum:
            raise BenchmarkInputError(f"{item_path}: golden earned_score cannot exceed maximum_score")

    outcomes = _list(page["outcomes"], f"{path}.outcomes")
    outcome_ids: set[str] = set()
    for index, item in enumerate(outcomes):
        item_path = f"{path}.outcomes[{index}]"
        outcome = _object(item, item_path)
        _required(outcome, ("id", "label"), item_path)
        outcome["id"] = _text(outcome["id"], f"{item_path}.id")
        outcome["label"] = _nullable_text(outcome["label"], f"{item_path}.label")
        if outcome["id"] in outcome_ids:
            raise BenchmarkInputError(f"{item_path}.id: duplicate outcome id")
        outcome_ids.add(outcome["id"])

    mappings = _list(page["mappings"], f"{path}.mappings")
    mapping_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(mappings):
        item_path = f"{path}.mappings[{index}]"
        mapping = _object(item, item_path)
        _required(mapping, ("question_id", "outcome_id", "weight_basis_points"), item_path)
        mapping["question_id"] = _question_id(mapping["question_id"], f"{item_path}.question_id")
        mapping["outcome_id"] = _text(mapping["outcome_id"], f"{item_path}.outcome_id")
        weight = mapping["weight_basis_points"]
        if weight is not None:
            _integer(weight, f"{item_path}.weight_basis_points", minimum=1)
            if weight > 10_000:
                raise BenchmarkInputError(f"{item_path}.weight_basis_points: must be <= 10000")
        key = (_normalize_question_id(mapping["question_id"]), mapping["outcome_id"])
        if key in mapping_keys:
            raise BenchmarkInputError(f"{item_path}: duplicate question/outcome mapping")
        mapping_keys.add(key)

    regions = _list(page["regions"], f"{path}.regions")
    region_keys: set[str] = set()
    for index, item in enumerate(regions):
        region = _validate_region(item, f"{path}.regions[{index}]", page_index, golden=role == "golden")
        if region["field_key"] in region_keys:
            raise BenchmarkInputError(f"{path}.regions[{index}].field_key: duplicate region key")
        region_keys.add(region["field_key"])

    if role == "golden":
        review_fields = _list(page["review_required_fields"], f"{path}.review_required_fields")
        normalized = [_text(item, f"{path}.review_required_fields[{i}]") for i, item in enumerate(review_fields)]
        if len(set(normalized)) != len(normalized):
            raise BenchmarkInputError(f"{path}.review_required_fields: duplicate field key")
        return page

    if page["parse_status"] not in ("ok", "failed"):
        raise BenchmarkInputError(f"{path}.parse_status: expected 'ok' or 'failed'")
    parse_error = page["parse_error"]
    if parse_error is not None:
        _text(parse_error, f"{path}.parse_error")
    confidence = _object(page["confidence_by_field"], f"{path}.confidence_by_field")
    for key, score in confidence.items():
        _text(key, f"{path}.confidence_by_field key")
        value = _finite_number(score, f"{path}.confidence_by_field.{key}")
        if not 0.0 <= value <= 1.0:
            raise BenchmarkInputError(f"{path}.confidence_by_field.{key}: must be in [0, 1]")
    issues = _list(page["review_issues"], f"{path}.review_issues")
    for index, item in enumerate(issues):
        item_path = f"{path}.review_issues[{index}]"
        issue = _object(item, item_path)
        _required(issue, ("field_key", "reason"), item_path)
        _text(issue["field_key"], f"{item_path}.field_key")
        if issue["reason"] not in REVIEW_REASONS:
            raise BenchmarkInputError(f"{item_path}.reason: unsupported review reason")
    if page["parse_status"] == "failed":
        if parse_error is None:
            raise BenchmarkInputError(f"{path}.parse_error: required when parse_status is failed")
        if (
            student["name"] is not None
            or student["number"] is not None
            or questions
            or outcomes
            or mappings
            or regions
            or confidence
        ):
            raise BenchmarkInputError(f"{path}: failed parses cannot contain extracted fields")
        if any(issue["field_key"] != "page" for issue in issues):
            raise BenchmarkInputError(
                f"{path}.review_issues: failed parses may only request page-level review"
            )
    elif parse_error is not None:
        raise BenchmarkInputError(f"{path}.parse_error: must be null when parse_status is ok")
    return page


def validate_document(value: Any, role: str) -> dict[str, Any]:
    if role not in ("golden", "predictions"):
        raise BenchmarkInputError("role must be 'golden' or 'predictions'")
    document = _object(value, role)
    _required(document, ("schema_version", "sessions"), role)
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise BenchmarkInputError(
            f"{role}.schema_version: expected integer {SCHEMA_VERSION}, got {version!r}"
        )
    if role == "predictions":
        _required(document, ("manifest",), role)
        _validate_manifest(document["manifest"])
    else:
        _required(document, ("corpus_type",), role)
        if document["corpus_type"] not in ("synthetic", "anonymized_exam_pages"):
            raise BenchmarkInputError(
                "golden.corpus_type: expected 'synthetic' or 'anonymized_exam_pages'"
            )
    sessions = _list(document["sessions"], f"{role}.sessions")
    if not sessions:
        raise BenchmarkInputError(f"{role}.sessions: at least one session is required")
    session_ids: set[str] = set()
    for session_index, item in enumerate(sessions):
        path = f"{role}.sessions[{session_index}]"
        session = _object(item, path)
        _required(session, ("session_id", "pages"), path)
        session["session_id"] = _text(session["session_id"], f"{path}.session_id")
        if session["session_id"] in session_ids:
            raise BenchmarkInputError(f"{path}.session_id: duplicate session id")
        session_ids.add(session["session_id"])
        pages = _list(session["pages"], f"{path}.pages")
        if not pages:
            raise BenchmarkInputError(f"{path}.pages: at least one page is required")
        page_indexes: set[int] = set()
        session_question_ids: set[str] = set()
        session_outcome_ids: set[str] = set()
        for page_index, page in enumerate(pages):
            normalized = _validate_page(page, f"{path}.pages[{page_index}]", role)
            if normalized["page_index"] in page_indexes:
                raise BenchmarkInputError(f"{path}.pages[{page_index}].page_index: duplicate page index")
            page_indexes.add(normalized["page_index"])
            session_question_ids.update(
                _normalize_question_id(question["id"]) for question in normalized["questions"]
            )
            session_outcome_ids.update(outcome["id"] for outcome in normalized["outcomes"])
        if role == "golden":
            for page_index, page in enumerate(pages):
                for mapping_index, mapping in enumerate(page["mappings"]):
                    question_id = _normalize_question_id(mapping["question_id"])
                    if question_id not in session_question_ids or mapping["outcome_id"] not in session_outcome_ids:
                        raise BenchmarkInputError(
                            f"{path}.pages[{page_index}].mappings[{mapping_index}]: "
                            "golden mapping references an unknown question/outcome in the session"
                        )
    return document


def _index_pages(document: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (session["session_id"], page["page_index"]): page
        for session in document["sessions"]
        for page in session["pages"]
    }


def _questions(page: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return (
        {}
        if page is None
        else {_normalize_question_id(item["id"]): item for item in page["questions"]}
    )


def _outcomes(page: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {} if page is None else {item["id"]: item for item in page["outcomes"]}


def _mappings(page: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if page is None:
        return {}
    return {
        (_normalize_question_id(item["question_id"]), item["outcome_id"]): item
        for item in page["mappings"]
    }


def _field_values(page: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in ("name", "number"):
        value = page["student"][field]
        if value is not None:
            values[f"student.{field}"] = value
    for question in page["questions"]:
        normalized_id = _normalize_question_id(question["id"])
        for field in ("id", "earned_score", "maximum_score"):
            value = question[field]
            if value is not None:
                values[f"question:{normalized_id}.{field}"] = value
    for outcome in page["outcomes"]:
        values[f"outcome:{outcome['id']}.id"] = outcome["id"]
        if outcome["label"] is not None:
            values[f"outcome:{outcome['id']}.label"] = outcome["label"]
    for mapping in page["mappings"]:
        question_id = _normalize_question_id(mapping["question_id"])
        prefix = f"mapping:{question_id}->{mapping['outcome_id']}"
        values[prefix] = (question_id, mapping["outcome_id"])
        if mapping["weight_basis_points"] is not None:
            values[f"{prefix}.allocation"] = mapping["weight_basis_points"]
    return values


def _score_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return Decimal(left) == Decimal(right)


def _box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = left["bbox"], right["bbox"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    intersection_width = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    intersection_height = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    intersection = intersection_width * intersection_height
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _expected_review_fields(gold: dict[str, Any]) -> set[str]:
    return set(gold["review_required_fields"])


def _page_is_perfect(gold: dict[str, Any], prediction: dict[str, Any] | None) -> bool:
    if prediction is None or prediction["parse_status"] != "ok":
        return False
    if gold["student"] != prediction["student"]:
        return False
    gold_questions, predicted_questions = _questions(gold), _questions(prediction)
    if set(gold_questions) != set(predicted_questions):
        return False
    for question_id, expected in gold_questions.items():
        actual = predicted_questions[question_id]
        if not _score_equal(expected["earned_score"], actual["earned_score"]):
            return False
        if not _score_equal(expected["maximum_score"], actual["maximum_score"]):
            return False
    gold_outcomes, predicted_outcomes = _outcomes(gold), _outcomes(prediction)
    if set(gold_outcomes) != set(predicted_outcomes):
        return False
    if any(gold_outcomes[key]["label"] != predicted_outcomes[key]["label"] for key in gold_outcomes):
        return False
    if set(_mappings(gold)) != set(_mappings(prediction)):
        return False
    if any(
        _mappings(gold)[key]["weight_basis_points"] != _mappings(prediction)[key]["weight_basis_points"]
        for key in _mappings(gold)
    ):
        return False
    expected_reviews = _expected_review_fields(gold)
    actual_reviews = {issue["field_key"] for issue in prediction["review_issues"]}
    return expected_reviews == actual_reviews


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "rate": None if total == 0 else round(correct / total, 6),
    }


def evaluate(golden: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    golden_pages = _index_pages(golden)
    predicted_pages = _index_pages(predictions)
    total_pages = len(golden_pages)
    parse_successes = 0
    totals: dict[str, list[int]] = {
        name: [0, 0]
        for name in (
            "student_number_exact_match",
            "student_name_exact_match",
            "question_id_exact_match",
            "score_exact_match",
            "maximum_score_exact_match",
            "outcome_id_exact_match",
            "outcome_label_exact_match",
            "question_to_outcome_mapping_accuracy",
            "allocation_exact_match",
            "field_coverage",
            "review_precision",
            "review_recall",
            "bounding_box_coverage",
            "bounding_box_validity",
            "confidence_coverage",
        )
    }
    review_true_positive = review_predicted_count = 0
    review_false_positive = review_false_negative = 0
    bbox_iou_sum = bbox_iou_count = 0.0
    bbox_expected_count = bbox_covered_count = 0
    invalid_regions = predicted_regions = 0
    page_perfect = 0
    per_session_perfect: dict[str, bool] = {}

    def add(name: str, correct: int, count: int = 1) -> None:
        totals[name][0] += correct
        totals[name][1] += count

    for key, gold_page in golden_pages.items():
        predicted = predicted_pages.get(key)
        parsed = predicted is not None and predicted["parse_status"] == "ok"
        parse_successes += int(parsed)
        gold_fields = _field_values(gold_page)
        predicted_fields = _field_values(predicted) if parsed else {}
        add("field_coverage", len(gold_fields.keys() & predicted_fields.keys()), len(gold_fields))
        add("confidence_coverage", len(gold_fields.keys() & (predicted.get("confidence_by_field", {}) if parsed else {}).keys()), len(gold_fields))

        for field, metric in (("number", "student_number_exact_match"), ("name", "student_name_exact_match")):
            expected = gold_page["student"][field]
            if expected is not None:
                actual = predicted["student"][field] if parsed else None
                add(metric, int(actual == expected))

        expected_questions, actual_questions = _questions(gold_page), _questions(predicted) if parsed else {}
        if expected_questions or actual_questions:
            add("question_id_exact_match", int(set(expected_questions) == set(actual_questions)))
        for question_id, expected in expected_questions.items():
            actual = actual_questions.get(question_id, {})
            for field, metric in (("earned_score", "score_exact_match"), ("maximum_score", "maximum_score_exact_match")):
                if expected[field] is not None:
                    add(metric, int(parsed and _score_equal(expected[field], actual.get(field))))

        expected_outcomes, actual_outcomes = _outcomes(gold_page), _outcomes(predicted) if parsed else {}
        if expected_outcomes or actual_outcomes:
            add("outcome_id_exact_match", int(set(expected_outcomes) == set(actual_outcomes)))
        for outcome_id, expected in expected_outcomes.items():
            if expected["label"] is not None:
                actual = actual_outcomes.get(outcome_id, {})
                add("outcome_label_exact_match", int(parsed and actual.get("label") == expected["label"]))

        expected_mappings, actual_mappings = _mappings(gold_page), _mappings(predicted) if parsed else {}
        union = set(expected_mappings) | set(actual_mappings)
        if union:
            add("question_to_outcome_mapping_accuracy", len(set(expected_mappings) & set(actual_mappings)), len(union))
        for mapping_key, expected in expected_mappings.items():
            if expected["weight_basis_points"] is not None:
                actual = actual_mappings.get(mapping_key, {})
                add("allocation_exact_match", int(parsed and actual.get("weight_basis_points") == expected["weight_basis_points"]))

        expected_review = _expected_review_fields(gold_page)
        predicted_review = (
            {issue["field_key"] for issue in predicted["review_issues"]} if predicted is not None else set()
        )
        review_true_positive += len(expected_review & predicted_review)
        review_predicted_count += len(predicted_review)
        review_false_positive += len(predicted_review - expected_review)
        review_false_negative += len(expected_review - predicted_review)

        expected_regions = {region["field_key"]: region for region in gold_page["regions"]}
        actual_regions = {region["field_key"]: region for region in predicted["regions"]} if parsed else {}
        bbox_expected_count += len(expected_regions)
        bbox_covered_count += len(expected_regions.keys() & actual_regions.keys())
        for field_key in expected_regions.keys() & actual_regions.keys():
            expected_region, actual_region = expected_regions[field_key], actual_regions[field_key]
            if _valid_region(actual_region, gold_page["page_index"]):
                bbox_iou_sum += _box_iou(expected_region, actual_region)
                bbox_iou_count += 1
        if parsed:
            for region in predicted["regions"]:
                predicted_regions += 1
                if _valid_region(region, gold_page["page_index"]):
                    pass
                else:
                    invalid_regions += 1

        is_perfect = _page_is_perfect(gold_page, predicted)
        page_perfect += int(is_perfect)
        per_session_perfect[key[0]] = per_session_perfect.get(key[0], True) and is_perfect

    for session in golden["sessions"]:
        session_id = session["session_id"]
        expected_indexes = {page["page_index"] for page in session["pages"]}
        predicted_indexes = {
            predicted_page["page_index"]
            for prediction_session in predictions["sessions"]
            if prediction_session["session_id"] == session_id
            for predicted_page in prediction_session["pages"]
        }
        if expected_indexes != predicted_indexes:
            per_session_perfect[session_id] = False

    for name, (correct, count) in totals.items():
        if name == "review_precision":
            totals[name] = [review_true_positive, review_predicted_count]
        elif name == "review_recall":
            totals[name] = [review_true_positive, review_true_positive + review_false_negative]
        elif name == "bounding_box_coverage":
            totals[name] = [bbox_covered_count, bbox_expected_count]
        elif name == "bounding_box_validity":
            totals[name] = [predicted_regions - invalid_regions, predicted_regions]
    parsed_metric = _metric(parse_successes, total_pages)
    session_count = len(golden["sessions"])
    unexpected_sessions = len({session["session_id"] for session in predictions["sessions"]} - {session["session_id"] for session in golden["sessions"]})
    expected_keys = set(golden_pages)
    unexpected_pages = len(set(predicted_pages) - expected_keys)

    result_metrics = {name: _metric(*counts) for name, counts in totals.items()}
    result_metrics["schema_parse_rate"] = parsed_metric
    result_metrics["schema_parse_failures"] = total_pages - parse_successes
    result_metrics["bounding_box_mean_iou"] = {
        "mean_iou": None if bbox_iou_count == 0 else round(bbox_iou_sum / bbox_iou_count, 6),
        "matched_valid_boxes": int(bbox_iou_count),
    }
    result_metrics["bounding_box_invalid_prediction_count"] = invalid_regions
    result_metrics["review_counts"] = {
        "true_positive": review_true_positive,
        "false_positive": review_false_positive,
        "false_negative": review_false_negative,
    }
    result_metrics["review_false_negative_count"] = review_false_negative
    result_metrics["whole_page_perfect_extraction"] = _metric(page_perfect, total_pages)
    session_perfect_count = sum(per_session_perfect.get(session["session_id"], False) for session in golden["sessions"])
    result_metrics["whole_session_perfect_extraction"] = _metric(session_perfect_count, session_count)

    corpus_type = golden.get("corpus_type", "anonymized_exam_pages")
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "corpus_type": corpus_type,
        "gate_status": "NOT_A_REAL_PAGE_BENCHMARK" if corpus_type == "synthetic" else "METRICS_ONLY_NO_THRESHOLD_SET",
        "normalization": {
            "question_id": "trim + ASCII A-Z case-fold + strip wrapping brackets and S/Soru/Q prefix (domain policy)",
            "student_number": "exact string; no normalization",
            "scores": "exact decimal value",
        },
        "manifest": predictions["manifest"],
        "counts": {
            "sessions": session_count,
            "pages": total_pages,
            "parsed_pages": parse_successes,
            "unexpected_prediction_sessions": unexpected_sessions,
            "unexpected_prediction_pages": unexpected_pages,
            "predicted_regions": predicted_regions,
        },
        "metrics": result_metrics,
    }


def _reject_json_constant(value: str) -> None:
    raise BenchmarkInputError(f"non-finite JSON number is not allowed: {value}")


def _read_document(path: Path, role: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, BenchmarkInputError) as exc:
        raise BenchmarkInputError(f"{path}: {exc}") from exc
    return validate_document(parsed, role)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, help="versioned ground-truth JSON")
    parser.add_argument("--predictions", type=Path, help="versioned prediction JSON")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path instead of stdout")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="evaluate the bundled perfect and deliberately faulty synthetic fixtures",
    )
    args = parser.parse_args(argv)
    try:
        if args.synthetic:
            if args.golden or args.predictions:
                parser.error("--synthetic cannot be combined with --golden or --predictions")
            fixtures = Path(__file__).with_name("fixtures")
            report = {
                "schema_version": SCHEMA_VERSION,
                "run_type": "synthetic_fixture_validation",
                "runs": {},
            }
            for name, golden_name, prediction_name in (
                (
                    "perfect",
                    "golden.perfect.synthetic.json",
                    "predictions.perfect.synthetic.json",
                ),
                ("deliberately_faulty", "golden.synthetic.json", "predictions.synthetic.json"),
            ):
                golden = _read_document(fixtures / golden_name, "golden")
                predictions = _read_document(fixtures / prediction_name, "predictions")
                report["runs"][name] = evaluate(golden, predictions)
        else:
            if not args.golden or not args.predictions:
                parser.error("--golden and --predictions are required unless --synthetic is used")
            golden = _read_document(args.golden, "golden")
            predictions = _read_document(args.predictions, "predictions")
            report = evaluate(golden, predictions)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
        return 0
    except BenchmarkInputError as exc:
        print(f"benchmark input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
