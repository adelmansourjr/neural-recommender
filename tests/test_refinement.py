"""Numerical, provenance, and publication regressions for the format 3 upgrade."""

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neural_recommender.als import ALSConfig, als_scores, fit_implicit_als, fold_in
from neural_recommender.api import create_app
from neural_recommender.audit import audit_serving
from neural_recommender.bundles import (
    atomic_savez,
    atomic_torch_save,
    file_hash,
    publish_bundle,
    resolve_bundle,
    rollback_bundle,
    verify_manifest,
)
from neural_recommender.conditional import combine_scores, history_bucket, validate_selector
from neural_recommender.data import load_dataset, split_users
from neural_recommender.diversity_evaluation import evaluate_diversity
from neural_recommender.leakage_audit import audit_leakage
from neural_recommender.optimization import OptimizationConfig, optimize
from neural_recommender.refinement import prepare_refinement_protocol, refine
from neural_recommender.service import RecommenderService


@pytest.fixture
def previous_upgrade(tmp_path):
    """A tiny real trained three-seed bundle; no test relies on installed artifacts."""
    data = tmp_path / "movies.zip"
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            "ml-1m/movies.dat",
            "\n".join(f"{index + 1}::Movie {index}::Drama" for index in range(24)),
        )
        archive.writestr(
            "ml-1m/ratings.dat",
            "\n".join(
                f"{user}::{(position + user) % 24 + 1}::5::{1000 + position}"
                for user in range(1, 61)
                for position in range(13)
            ),
        )
    _, ratings = load_dataset(data)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "data_sha256": file_hash(data),
                "config": {"seed": 42},
                "test_users": split_users(ratings, 42)["test"][:1].tolist(),
            }
        )
    )
    artifact = tmp_path / "artifacts"
    previous = tmp_path / "upgrade_protocol.json"
    optimize(
        data,
        artifact,
        tmp_path / "upgrade_report.json",
        previous,
        pilot,
        OptimizationConfig(
            epochs=1, train_users=10, evaluation_users=2, seeds=(7, 19, 42), threads=1
        ),
    )
    return data, artifact, previous, pilot, ratings


def test_als_fold_in_matches_weighted_normal_equations():
    factors = np.array([[1.0, 0.2], [0.5, 1.0], [0.2, 0.5]])
    history = np.array([0, 2])
    confidence = np.eye(3)
    confidence[history, history] += 10
    preference = np.array([1.0, 0, 1.0])
    expected = np.linalg.solve(
        factors.T @ confidence @ factors + 2 * np.eye(2), factors.T @ confidence @ preference
    )
    np.testing.assert_allclose(fold_in(factors, history, 2, 10), expected)
    np.testing.assert_allclose(als_scores(factors, history, 2, 10), factors @ expected)


@pytest.mark.parametrize(
    ("regularization", "alpha", "gram"),
    [
        (float("nan"), 10, None),
        (2, float("nan"), None),
        (0, 10, None),
        (2, 10, np.full((2, 2), np.nan)),
        (2, 10, np.eye(3)),
        (2, 10, np.array([[1, 2], [0, 1]])),
    ],
)
def test_als_rejects_invalid_solver_inputs(regularization, alpha, gram):
    with pytest.raises(ValueError):
        fold_in(np.eye(2), np.array([0]), regularization, alpha, gram)


def test_als_fits_only_supplied_interactions_and_is_repeatable():
    matrix = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    config = ALSConfig(factors=2, iterations=3, seed=7)
    first, report = fit_implicit_als(matrix, config)
    second, _ = fit_implicit_als(matrix, config)
    np.testing.assert_array_equal(first, second)
    assert report["positive_interactions"] == 6
    assert als_scores(first, np.array([0]), 10, 10)[1] > als_scores(first, np.array([0]), 10, 10)[2]
    with pytest.raises(ValueError):
        fit_implicit_als(np.array([[1, -1], [0, 1]]), config)


def test_conditional_selection_is_explicit_and_strict():
    assert [history_bucket(length) for length in (1, 3, 4, 7, 8, 20)] == [
        "1",
        "1",
        "5",
        "5",
        "10",
        "10",
    ]
    with pytest.raises(ValueError):
        history_bucket(0)
    with pytest.raises(ValueError):
        validate_selector({"1": {"name": "x", "weights": {"ease": 1}}}, {"ease"})
    with pytest.raises(ValueError):
        combine_scores({"ease": np.zeros(2)}, {"recursive": 1})


def test_protocol_freezes_reserve_membership_and_excludes_both_generations(
    previous_upgrade, tmp_path
):
    data, _, previous, pilot, ratings = previous_upgrade
    protocol_path = tmp_path / "new_protocol.json"
    protocol = prepare_refinement_protocol(
        ratings, file_hash(data), previous, pilot, protocol_path, 2
    )
    prior = json.loads(previous.read_text())
    excluded = set(prior["reserve_test_users"]) | set(json.loads(pilot.read_text())["test_users"])
    assert excluded.isdisjoint(protocol["reserve_test_users"])
    assert set(protocol["train_users"]).isdisjoint(protocol["reserve_test_users"])
    assert set(protocol["validation_users"]).isdisjoint(protocol["reserve_test_users"])
    assert protocol["available_reserve_users"] == 6
    assert (
        prepare_refinement_protocol(ratings, file_hash(data), previous, pilot, protocol_path, 2)
        == protocol
    )
    with pytest.raises(ValueError, match="changed"):
        prepare_refinement_protocol(ratings, file_hash(data), previous, pilot, protocol_path, 3)


def test_refinement_publish_roundtrip_modes_and_rollback(previous_upgrade, tmp_path):
    data, artifact, previous, pilot, _ = previous_upgrade
    report = refine(
        data,
        artifact,
        tmp_path / "protocol.json",
        tmp_path / "report.json",
        previous,
        pilot,
        reserve_users=2,
    )
    assert set(report["history_selector"]) == {"1", "5", "10"}
    service = RecommenderService(data, artifact)
    assert service.format_version == 3
    assert service.bundle_dir != artifact
    for count in (1, 5, 10):
        history = list(range(1, count + 1))
        choice = service.selection("accuracy", count)
        assert choice["selected_ranker"] == report["history_selector"][str(count)]["name"]
        assert service.recommend(history, 5, "accuracy") == service.recommend(
            history, 5, "adaptive"
        )
    for mode in (
        "accuracy",
        "adaptive",
        "blend",
        "als",
        "discovery",
        "hybrid",
        "neural",
        "ease",
        "item_knn",
        "popularity",
    ):
        predictions = service.recommend([1, 2], 5, mode)
        assert len(predictions) == 5
        assert not {1, 2} & {row["movie_id"] for row in predictions}
        assert np.isfinite([row["score"] for row in predictions]).all()
    client = TestClient(create_app(data, artifact))
    response = client.post("/recommend", json={"movie_ids": [1], "mode": "accuracy"})
    assert response.status_code == 200
    assert response.json()["selector_history_bucket"] == "1"
    assert client.get("/model-info").json()["history_selector"] == report["history_selector"]
    assert client.get("/health").json()["format_version"] == 3
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.post("/recommend", json={"movie_ids": list(range(1, 22))}).status_code == 422
    original = service.recommend([1, 2], 10, "accuracy")
    assert original == service.recommend([1, 2], 10, "accuracy", 1.0)
    assert (
        client.post("/recommend", json={"movie_ids": [1, 2]}).json()["recommendations"] == original
    )
    varied = client.post("/recommend", json={"movie_ids": [1, 2], "relevance_weight": 0.5})
    assert varied.status_code == 200
    assert varied.json()["reranking"]["enabled"]
    assert varied.json()["reranking"]["candidate_pool"] == 100
    assert not {1, 2} & {row["movie_id"] for row in varied.json()["recommendations"]}
    for invalid_weight in (True, "0.5", -0.1, 1.1):
        assert (
            client.post(
                "/recommend", json={"movie_ids": [1], "relevance_weight": invalid_weight}
            ).status_code
            == 422
        )
    # Requests share frozen model state. Different rankers must remain deterministic
    # when FastAPI's worker pool executes them concurrently.
    requests = [
        (list(range(1, count + 1)), mode)
        for count in (1, 5, 10)
        for mode in ("accuracy", "blend", "neural", "als")
    ]
    expected = [service.recommend(history, 10, mode) for history, mode in requests]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(
            executor.map(lambda request: service.recommend(request[0], 10, request[1]), requests)
        )
    assert actual == expected
    choice = service.selection("accuracy", 5)
    choice["weights"].clear()
    assert service.selection("accuracy", 5)["weights"]
    audit = audit_serving(data, artifact, tmp_path / "report.json", tmp_path / "audit.json")
    assert audit["status"] == "passed"
    assert audit["histories"]["5"]["mode_rankings_checked"] == 20
    diversity_report = evaluate_diversity(
        data,
        artifact,
        tmp_path / "report.json",
        tmp_path / "diversity-protocol.json",
        tmp_path / "diversity.json",
    )
    assert set(diversity_report["results"]) == {"1.0", "0.9", "0.75", "0.5", "0.0"}
    assert diversity_report["results"]["1.0"]["ndcg_at_10"]["mean"] == pytest.approx(
        report["reserve_test"]["5"]["adaptive"]["ndcg_at_10"]["mean"]
    )
    assert diversity_report["results"]["1.0"]["paired_ndcg_minus_default"]["mean"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_diversity(
            data,
            artifact,
            tmp_path / "report.json",
            tmp_path / "diversity-protocol.json",
            tmp_path / "diversity.json",
        )
    frozen = json.loads((tmp_path / "report.json").read_text())
    frozen["reserve_test"]["5"]["popularity"]["ndcg_at_10"]["mean"] += 0.01
    (tmp_path / "changed-report.json").write_text(json.dumps(frozen))
    with pytest.raises(AssertionError):
        audit_serving(data, artifact, tmp_path / "changed-report.json", tmp_path / "failure.json")
    assert not (tmp_path / "failure.json").exists()
    original = artifact / "rankers.npz"
    baseline_digest = file_hash(original)
    assert file_hash(service.bundle_dir / "rankers.npz") == baseline_digest
    # Replacing a legacy file must not mutate the new hard-linked generation.
    with np.load(original) as archive:
        arrays = {name: archive[name] for name in archive.files}
    atomic_savez(original, **arrays)
    assert file_hash(service.bundle_dir / "rankers.npz") == baseline_digest
    rollback_bundle(artifact)
    assert not (artifact / "active.json").exists()
    assert RecommenderService(data, artifact).format_version == 2


def test_failed_publish_does_not_change_active_pointer(previous_upgrade, tmp_path):
    data, artifact, _, _, _ = previous_upgrade
    service = RecommenderService(data, artifact)
    metadata = {**service.metadata, "format_version": 3, "seeds": [7, 19, 42]}
    (artifact / "rankers.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash"):
        publish_bundle(artifact, artifact, metadata, np.zeros((24, 16)))
    assert not (artifact / "active.json").exists()
    assert (
        not list((artifact / "bundles").glob("v3-*")) if (artifact / "bundles").exists() else True
    )


def test_leakage_audit_reconstructs_a_real_fixture_bundle(previous_upgrade, tmp_path):
    data, artifact, upgrade, pilot, _ = previous_upgrade
    reports = tmp_path / "receipts"
    reports.mkdir()
    for source, target in (
        (upgrade, "upgrade_protocol.json"),
        (pilot, "initial_pilot_metadata.json"),
        (tmp_path / "upgrade_report.json", "optimized_benchmark.json"),
    ):
        (reports / target).write_bytes(source.read_bytes())
    refine(
        data,
        artifact,
        reports / "refinement_protocol.json",
        reports / "refined_benchmark.json",
        reports / "upgrade_protocol.json",
        reports / "initial_pilot_metadata.json",
        reserve_users=2,
    )
    output = tmp_path / "leakage.json"
    result = audit_leakage(data, artifact, reports, output)
    assert result["status"] == "passed_recorded_user_isolation_checks"
    assert result["fitting"]["held_out_poisoning_preserves_matrix_and_example_hashes"]
    assert result["membership"]["poisoned_held_out_users"] == 7
    with pytest.raises(ValueError, match="new output"):
        audit_leakage(data, artifact, reports, output)


def publication_metadata(artifact):
    """Complete new-generation metadata reusing the fixture's frozen baselines."""
    metadata = json.loads((artifact / "metadata.json").read_text())
    selector = {
        bucket: {"name": "popularity", "weights": {"popularity": 1.0}}
        for bucket in ("1", "5", "10")
    }
    return {
        **metadata,
        "format_version": 3,
        "als_config": {"factors": 16},
        "history_selector": selector,
        "history_blends": selector,
    }


def test_publication_rejects_source_replaced_after_initial_verification(
    previous_upgrade, monkeypatch
):
    """A fresh hash must not silently bless a different source model generation."""
    from neural_recommender import bundles

    _, artifact, _, _, _ = previous_upgrade
    metadata = publication_metadata(artifact)
    original_link = bundles.os.link

    def replace_then_link(source, target):
        if source.name == "rankers.npz":
            replacement = source.with_suffix(".changed")
            replacement.write_bytes(b"replacement after verification")
            replacement.replace(source)
        original_link(source, target)

    monkeypatch.setattr(bundles.os, "link", replace_then_link)
    with pytest.raises(ValueError, match="source|hash"):
        publish_bundle(artifact, artifact, metadata, np.zeros((24, 16)))
    assert not (artifact / "active.json").exists()
    assert not list((artifact / "bundles").glob("v3-*"))
    assert not (artifact / ".bundle-writer.lock").exists()


def test_publish_and_rollback_cannot_interleave_pointer_updates(previous_upgrade, monkeypatch):
    from neural_recommender import bundles

    data, artifact, _, _, _ = previous_upgrade
    metadata = publication_metadata(artifact)
    copying, release = Event(), Event()
    original_link = bundles.os.link

    def pause_link(source, target):
        if not copying.is_set():
            copying.set()
            assert release.wait(timeout=5), "Concurrent publication test was not released"
        original_link(source, target)

    monkeypatch.setattr(bundles.os, "link", pause_link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(publish_bundle, artifact, artifact, metadata, np.zeros((24, 16)))
        try:
            assert copying.wait(timeout=5)
            with pytest.raises(ValueError, match="writer"):
                publish_bundle(artifact, artifact, metadata, np.zeros((24, 16)))
            with pytest.raises(ValueError, match="writer"):
                rollback_bundle(artifact)
            assert not (artifact / "active.json").exists()
        finally:
            release.set()
        published = pending.result(timeout=5)
    assert resolve_bundle(artifact) == published
    assert RecommenderService(data, artifact).format_version == 3
    rollback_bundle(artifact)
    assert RecommenderService(data, artifact).format_version == 2


@pytest.mark.parametrize(
    "value", [{"bundle": "../outside"}, {"bundle": None}, {"unexpected": True}]
)
def test_active_pointer_cannot_escape_bundle_directory(tmp_path, value):
    (tmp_path / "active.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        resolve_bundle(tmp_path)


@pytest.mark.parametrize("version", [True, "3", 3.0, None])
def test_checkpoint_format_version_must_be_an_actual_integer(tmp_path, version):
    with pytest.raises(ValueError, match="integer"):
        verify_manifest(tmp_path, {"format_version": version})


@pytest.mark.parametrize("metadata", [[], None, "format3"])
def test_manifest_requires_json_object(tmp_path, metadata):
    with pytest.raises(ValueError, match="object"):
        verify_manifest(tmp_path, metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config", None),
        ("fusion_weights", [0.5, 0.5]),
        ("fusion_weights", [True, 0, 0]),
        ("fusion_weights", [float("nan"), 0, 0]),
        ("discovery_penalty", None),
        ("accuracy_model", "recursive"),
    ],
)
def test_service_rejects_malformed_settings_before_inference(previous_upgrade, field, value):
    data, artifact, _, _, _ = previous_upgrade
    path = artifact / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        RecommenderService(data, artifact)


@pytest.mark.parametrize("value", [99.0, float("nan")])
def test_hash_valid_checkpoint_cannot_override_catalog_features(previous_upgrade, value):
    data, artifact, _, _, _ = previous_upgrade
    model = artifact / "neural_0.pt"
    state = torch.load(model, weights_only=True)
    state["genres"][1, 0] = value
    atomic_torch_save(state, model)
    path = artifact / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["artifact_hashes"][model.name] = file_hash(model)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="genre features"):
        RecommenderService(data, artifact)


def test_hash_valid_ranker_archive_requires_real_numbers(previous_upgrade):
    data, artifact, _, _, _ = previous_upgrade
    path = artifact / "rankers.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["ease"] = arrays["ease"].astype(str)
    atomic_savez(path, **arrays)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_hashes"][path.name] = file_hash(path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="real numbers"):
        RecommenderService(data, artifact)


def test_als_rejects_overflow_and_non_positive_systems_before_returning_scores():
    with pytest.raises(ValueError, match="ALS"):
        fold_in(np.full((2, 2), 1e308), np.array([0]), 2, 10, np.eye(2))
    with pytest.raises(ValueError, match="positive-definite"):
        fold_in(np.eye(2), np.array([0]), 2, 10, -100 * np.eye(2))


@pytest.mark.parametrize("weight", ["1", True, float("nan"), complex(1, 0)])
def test_selector_weights_reject_ambiguous_and_nonfinite_numbers(weight):
    selector = {
        bucket: {"name": "ease", "weights": {"ease": weight}} for bucket in ("1", "5", "10")
    }
    with pytest.raises(ValueError):
        validate_selector(selector, {"ease"})


@pytest.mark.parametrize(
    ("movies", "message"),
    [
        ("\n".join(f"{i}::Movie {i}::Drama" for i in range(1, 10002)), "budget"),
        ("1::Movie::" + "|".join(f"genre{i}" for i in range(129)), "feature budget"),
        ("1::::Drama", "titles"),
        ("1::Movie::", "feature budget"),
    ],
)
def test_catalog_allocation_and_display_fields_are_bounded(tmp_path, movies, message):
    path = tmp_path / "bad-data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-1m/movies.dat", movies)
        archive.writestr("ml-1m/ratings.dat", "1::1::5::1000")
    with pytest.raises(ValueError, match=message):
        load_dataset(path)
