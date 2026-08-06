import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from loguru import logger

from app.core.config import CorpusName, config
from app.evaluation.answer_eval import run_answer_evaluation
from app.evaluation.retrieval_eval import run_retrieval_evaluation


DEFAULT_BASELINE_PATH = Path(
    "data/eval/release_baseline_v1.json"
)
DEFAULT_REPORT_PATH = Path(
    "data/eval/release_check_results.json"
)


def run_release_check(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    baseline = _load_baseline(baseline_path)
    expected_model = baseline.get("model")

    if expected_model != config.ollama_model:
        raise RuntimeError(
            "Model Ollama nie zgadza się z baseline: "
            f"oczekiwano {expected_model!r}, "
            f"uruchomiono {config.ollama_model!r}."
        )

    observed_profiles: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for corpus in ("v1", "v2"):
        corpus_name = cast(CorpusName, corpus)
        profile = config.activate_corpus(corpus_name)

        print()
        print("#" * 80)
        print(f"RELEASE CHECK — CORPUS {corpus.upper()}")
        print("#" * 80)

        logger.info(
            "Release gate started | corpus={} | collection={}",
            corpus,
            profile.qdrant_collection,
        )

        retrieval_summary = run_retrieval_evaluation(
            dataset_path=profile.retrieval_eval_file
        )
        answer_summary = run_answer_evaluation(
            dataset_path=profile.answer_eval_file,
            report_path=profile.answer_eval_results_file,
        )

        observed = {
            "retrieval": asdict(retrieval_summary),
            "answers": asdict(answer_summary),
        }
        observed_profiles[corpus] = observed

        expected_profile = baseline["profiles"].get(corpus)

        if not isinstance(expected_profile, dict):
            failures.append(
                f"{corpus}: brak profilu w baseline"
            )
            continue

        failures.extend(
            _compare_profile(
                corpus=corpus,
                observed=observed,
                expected=expected_profile,
            )
        )

    passed = not failures
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_version": baseline["baseline_version"],
        "baseline_path": str(baseline_path),
        "model": config.ollama_model,
        "passed": passed,
        "failures": failures,
        "profiles": observed_profiles,
    }
    _save_report(report, report_path)
    _print_final_summary(passed, failures, report_path)

    if not passed:
        raise RuntimeError(
            "Bramka wydania wykryła regresję. "
            "Sprawdź podsumowanie i raport JSON."
        )


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku baseline: {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    required_keys = {
        "baseline_version",
        "model",
        "profiles",
    }
    missing_keys = required_keys - set(data)

    if missing_keys:
        raise ValueError(
            "Nieprawidłowy plik baseline. Brak pól: "
            f"{sorted(missing_keys)}"
        )

    if not isinstance(data["profiles"], dict):
        raise ValueError(
            "Pole 'profiles' w baseline musi być obiektem."
        )

    return data


def _compare_profile(
    *,
    corpus: str,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    failures: list[str] = []

    for section in ("retrieval", "answers"):
        expected_section = expected.get(section)

        if not isinstance(expected_section, dict):
            failures.append(
                f"{corpus}.{section}: brak progów"
            )
            continue

        observed_section = observed[section]
        expected_cases = expected_section.get("total_cases")

        if observed_section["total_cases"] != expected_cases:
            failures.append(
                f"{corpus}.{section}.total_cases: "
                f"wynik {observed_section['total_cases']}, "
                f"baseline {expected_cases}"
            )

        failures.extend(
            _compare_metrics(
                prefix=f"{corpus}.{section}",
                observed=observed_section,
                thresholds=expected_section.get("minimum", {}),
                comparison="minimum",
            )
        )
        failures.extend(
            _compare_metrics(
                prefix=f"{corpus}.{section}",
                observed=observed_section,
                thresholds=expected_section.get("maximum", {}),
                comparison="maximum",
            )
        )

    return failures


def _compare_metrics(
    *,
    prefix: str,
    observed: dict[str, Any],
    thresholds: dict[str, Any],
    comparison: str,
) -> list[str]:
    failures: list[str] = []

    if not isinstance(thresholds, dict):
        return [f"{prefix}: nieprawidłowe progi {comparison}"]

    for metric, threshold in thresholds.items():
        value = observed.get(metric)

        if not isinstance(value, (int, float)):
            failures.append(
                f"{prefix}.{metric}: brak wartości liczbowej"
            )
            continue

        failed = (
            value < threshold
            if comparison == "minimum"
            else value > threshold
        )

        if failed:
            operator = ">=" if comparison == "minimum" else "<="
            failures.append(
                f"{prefix}.{metric}: wynik {value:.4f}, "
                f"wymagane {operator} {threshold:.4f}"
            )

    return failures


def _save_report(
    report: dict[str, Any],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Release gate report saved: {}", report_path)


def _print_final_summary(
    passed: bool,
    failures: list[str],
    report_path: Path,
) -> None:
    print()
    print("=" * 80)
    print("RELEASE CHECK — PODSUMOWANIE")
    print("=" * 80)
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Liczba regresji: {len(failures)}")

    for failure in failures:
        print(f"- {failure}")

    print(f"Raport: {report_path}")
