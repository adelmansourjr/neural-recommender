import hashlib
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neural_recommender.api import create_app
from neural_recommender.baselines import (
    ease_scores,
    fit_ease,
    fit_item_knn,
    fuse_scores,
    interaction_gram,
    interaction_matrix,
    knn_scores,
)
from neural_recommender.data import load_dataset, split_users, user_sequences
from neural_recommender.optimization import (
    OptimizationConfig,
    optimize,
    prepare_protocol,
    select_accuracy_model,
)
from neural_recommender.service import RecommenderService


@pytest.fixture
def fixture_data(tmp_path):
    path = tmp_path / "movies.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ml-1m/movies.dat",
            "\n".join(f"{index + 1}::Movie {index}::Drama" for index in range(24)),
        )
        archive.writestr(
            "ml-1m/ratings.dat",
            "\n".join(
                f"{user}::{(position + user) % 24 + 1}::5::{1000 + position}"
                for user in range(1, 21)
                for position in range(13)
            ),
        )
    _, ratings = load_dataset(path)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "config": {"seed": 42},
                "test_users": split_users(ratings, 42)["test"][:1].tolist(),
            }
        )
    )
    return path, pilot


def test_ease_has_zero_self_weights_and_respects_input_direction():
    gram = np.array([[4.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    weights = fit_ease(gram, 1)
    np.testing.assert_array_equal(np.diag(weights), np.zeros(3))
    assert weights[0, 1] > weights[0, 2]
    np.testing.assert_array_equal(ease_scores(weights, np.array([0])), weights[0])
    with pytest.raises(ValueError):
        fit_ease(gram, 0)


def test_item_knn_has_no_self_contribution():
    gram = np.array([[4.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    indices, values = fit_item_knn(gram, 2, neighbors=1)
    scores = knn_scores(indices, values, np.array([0]))
    assert scores[0] == 0
    assert scores[1] > 0
    assert scores[2] == 0


def test_fusion_respects_declared_weights():
    ease, neural, popularity = np.array([1, 3, 2]), np.array([3, 2, 1]), np.array([1, 1, 2])
    scores = fuse_scores(ease, neural, popularity, (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(np.argsort(scores), np.argsort(ease))
    with pytest.raises(ValueError):
        fuse_scores(ease, neural, popularity, (1, 1, 1))


def test_reserve_protocol_excludes_inspected_users_and_gram_uses_train_only(fixture_data, tmp_path):
    path, pilot = fixture_data
    _, ratings = load_dataset(path)
    config = OptimizationConfig(epochs=1, train_users=10, evaluation_users=2, seeds=(7, 19))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = prepare_protocol(ratings, digest, pilot, tmp_path / "protocol.json", config)
    assert set(protocol["reserve_test_users"]).isdisjoint(protocol["excluded_pilot_test_users"])
    assert set(protocol["reserve_test_users"]).isdisjoint(protocol["train_users"])
    assert set(protocol["reserve_test_users"]).isdisjoint(protocol["validation_users"])
    sequences, _ = user_sequences(ratings)
    matrix = interaction_matrix(sequences, np.array(protocol["train_users"]), 24)
    gram = interaction_gram(matrix)
    for user in protocol["reserve_test_users"]:
        sequences[user] = np.array([0])
    np.testing.assert_array_equal(
        gram, interaction_gram(interaction_matrix(sequences, np.array(protocol["train_users"]), 24))
    )
    with pytest.raises(ValueError, match="differs"):
        prepare_protocol(ratings, "changed", pilot, tmp_path / "protocol.json", config)


def test_optimized_training_serving_and_modes(fixture_data, tmp_path):
    path, pilot = fixture_data
    artifact = tmp_path / "artifacts"
    result = optimize(
        path,
        artifact,
        tmp_path / "report.json",
        tmp_path / "protocol.json",
        pilot,
        OptimizationConfig(epochs=1, train_users=10, evaluation_users=2, seeds=(7, 19), threads=1),
    )
    assert len(result["neural_learning_curves"]) == 2
    assert set(result["reserve_test"]) == {"1", "5", "10"}
    service = RecommenderService(path, artifact)
    assert service.format_version == 2
    for mode in ("accuracy", "discovery", "hybrid", "neural", "ease", "item_knn", "popularity"):
        output = service.recommend([1, 2], 5, mode)
        assert len(output) == 5
        assert {row["movie_id"] for row in output}.isdisjoint({1, 2})
    client = TestClient(create_app(path, artifact))
    assert client.get("/model-info").json()["format_version"] == 2
    winner = result["serving_model_selection"]["winner"]
    assert service.recommend([1, 2], 5, "accuracy") == service.recommend([1, 2], 5, winner)
    discovery = service.recommend([1, 2], 5, "discovery")
    assert [row["score"] for row in discovery] == sorted(
        (row["score"] for row in discovery), reverse=True
    )
    assert (
        client.post("/recommend", json={"movie_ids": [1], "mode": "discovery"}).status_code == 200
    )
    assert client.post("/recommend", json={"movie_ids": [1], "mode": "invented"}).status_code == 422
    rankers = artifact / "rankers.npz"
    rankers.write_bytes(rankers.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        RecommenderService(path, artifact)


def test_accuracy_selection_compares_all_candidates_without_test_scores():
    assert (
        select_accuracy_model({"popularity": 0.5, "hybrid": 0.4, "item_knn": 0.3}) == "popularity"
    )
    assert select_accuracy_model({"ease": 0.3, "neural": 0.4, "item_knn": 0.5}) == "item_knn"
    with pytest.raises(ValueError):
        select_accuracy_model({"hybrid": float("nan")})
