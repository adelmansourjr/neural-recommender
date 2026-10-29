import json
import zipfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neural_recommender.api import create_app
from neural_recommender.data import load_dataset, split_users, training_examples, user_sequences
from neural_recommender.metrics import rank, ranking_metrics, score_items
from neural_recommender.model import HistoryRecommender
from neural_recommender.service import RecommenderService
from neural_recommender.training import TrainConfig, train


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "movies.zip"
    movies = "\n".join(
        f"{index + 1}::Film {index}::{'Comedy' if index % 2 else 'Drama'}" for index in range(24)
    )
    rows = []
    for user in range(1, 21):
        for position in range(13):
            movie = (position + user) % 24 + 1
            rows.append(f"{user}::{movie}::5::{1000 + position}")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-1m/movies.dat", movies)
        archive.writestr("ml-1m/ratings.dat", "\n".join(rows))
    return path


def test_user_split_is_disjoint_and_repeatable(data_path):
    _, ratings = load_dataset(data_path)
    splits = split_users(ratings, 42)
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    for name, values in splits.items():
        np.testing.assert_array_equal(values, split_users(ratings, 42)[name])


def test_training_histories_cannot_contain_their_targets(data_path):
    _, ratings = load_dataset(data_path)
    sequences, _ = user_sequences(ratings)
    histories, targets, _ = training_examples(sequences, np.array([1]), history_size=5)
    assert all(
        target + 1 not in history for history, target in zip(histories, targets, strict=True)
    )


def test_ranking_excludes_seen_movies_and_has_known_metrics():
    selected = rank(np.array([9, 8, 7, 6]), np.array([0]), 2)
    np.testing.assert_array_equal(selected, [1, 2])
    recall, ndcg = ranking_metrics(selected, {1, 2}, 2)
    assert recall == ndcg == 1
    assert len(rank(np.array([1.0]), np.array([0]), 5)) == 0
    with pytest.raises(ValueError):
        rank(np.array([np.nan]), np.array([], dtype=int), 1)


def test_padding_does_not_change_user_representation():
    torch.manual_seed(0)
    model = HistoryRecommender(torch.eye(3), dimension=4)
    first = model.encode_history(torch.tensor([[1, 2]]))
    padded = model.encode_history(torch.tensor([[1, 2, 0, 0]]))
    torch.testing.assert_close(first, padded)


def test_training_checkpoint_and_api_integration(data_path, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    report = train(
        data_path,
        artifact_dir,
        tmp_path / "report.json",
        TrainConfig(
            epochs=2, dimension=8, batch_size=16, max_train_users=10, evaluation_users=2, threads=1
        ),
    )
    assert np.isfinite(report["learning_curve"][-1]["loss"])
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert set(metadata["train_users"]).isdisjoint(metadata["test_users"])
    service = RecommenderService(data_path, artifact_dir)
    recommendations = service.recommend([1, 2], 5)
    assert len(recommendations) == 5
    assert not {1, 2} & {item["movie_id"] for item in recommendations}
    client = TestClient(create_app(data_path, artifact_dir))
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert (
        client.post("/recommend", json={"movie_ids": [1, 2], "k": 5}).json()["recommendations"]
        == recommendations
    )
    assert client.post("/recommend", json={"movie_ids": [9999]}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": [1, 1]}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": []}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": [True]}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": ["1"]}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": [1], "unexpected": True}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": list(range(1, 22))}).status_code == 422
    assert client.get("/movies?q=Film%201").json()
    assert client.get("/movies?q=" + "a" * 101).status_code == 422
    with pytest.raises(ValueError, match="unique"):
        service.recommend([1, 1])
    original = data_path.read_bytes()
    data_path.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="Dataset differs"):
        RecommenderService(data_path, artifact_dir)
    data_path.write_bytes(original)
    checkpoint = artifact_dir / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        RecommenderService(data_path, artifact_dir)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        TrainConfig(epochs=0).validate()


def test_embedding_scores_are_finite_and_match_scalar_reference():
    rng = np.random.default_rng(42)
    items = rng.normal(size=(3883, 32)).astype(np.float32)
    user = rng.normal(size=32).astype(np.float32)
    with np.errstate(all="raise"):
        scores = score_items(items, user)
    expected = np.array(
        [sum(float(a) * float(b) for a, b in zip(row, user, strict=True)) for row in items]
    )
    np.testing.assert_allclose(scores, expected, atol=1e-5)
    with pytest.raises(ValueError, match="finite"):
        score_items(np.array([[np.nan]]), np.ones(1))


@pytest.mark.parametrize(
    "ratings",
    [
        "1::-1::5::1000",
        "1::999::5::1000",
        "1::1::6::1000",
        "1::1::5::1000\n1::1::4::1001",
    ],
)
def test_bad_ratings_fail_before_indexing(tmp_path, ratings):
    path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-1m/movies.dat", "1::Film::Drama")
        archive.writestr("ml-1m/ratings.dat", ratings)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_invalid_checkpoint_format_rejects_without_deserializing(data_path, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"format_version": 999}))
    with pytest.raises(ValueError, match="format"):
        RecommenderService(data_path, tmp_path)
