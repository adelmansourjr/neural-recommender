"""MovieLens ingestion and user-disjoint history/target construction."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .storage import download

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass(frozen=True)
class Catalog:
    ids: np.ndarray
    titles: tuple[str, ...]
    genres: np.ndarray
    genre_names: tuple[str, ...]


def fetch_dataset(directory: Path) -> Path:
    path = directory / "ml-1m.zip"
    download(URL, path, max_bytes=10_000_000)
    return path


def load_dataset(path: Path) -> tuple[Catalog, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        for name in ("ml-1m/movies.dat", "ml-1m/ratings.dat"):
            if archive.getinfo(name).file_size > 40_000_000:
                raise ValueError("Dataset member exceeds the decompressed size budget")
        movies = [
            line.split("::")
            for line in archive.read("ml-1m/movies.dat").decode("latin-1").splitlines()
        ]
        ratings = np.loadtxt(
            io.BytesIO(archive.read("ml-1m/ratings.dat")),
            delimiter=":",
            usecols=(0, 2, 4, 6),
            dtype=np.int64,
            ndmin=2,
        )
    if not movies or any(len(row) != 3 for row in movies) or not len(ratings):
        raise ValueError("Dataset needs well-formed movie records and ratings")
    if len(movies) > 10_000 or len(ratings) > 1_500_000:
        raise ValueError("Dataset exceeds the supported MovieLens-1M catalog/rating budget")
    if any(not title.strip() or len(title) > 500 for _, title, _ in movies):
        raise ValueError("Movie titles must be nonempty and at most 500 characters")
    ids = np.array([int(row[0]) for row in movies], dtype=np.int64)
    if len(set(ids)) != len(ids) or np.any(ids < 1) or ids.max() > 1_000_000:
        raise ValueError("Movie identifiers must be unique positive bounded integers")
    if np.any(ratings[:, :2] < 1) or np.any(ratings[:, 3] < 0):
        raise ValueError("Rating identifiers must be positive and timestamps nonnegative")
    if len(np.unique(ratings[:, :2], axis=0)) != len(ratings):
        raise ValueError("Duplicate user/movie ratings are not supported")
    genres = tuple(sorted({genre for _, _, text in movies for genre in text.split("|")}))
    # Bound the dense feature allocation before constructing catalog metadata.
    if len(genres) > 128 or any(not genre.strip() or len(genre) > 100 for genre in genres):
        raise ValueError("Catalog genres exceed the supported feature budget")
    features = np.zeros((len(movies), len(genres)), dtype=np.float32)
    for index, (_, _, names) in enumerate(movies):
        for genre in names.split("|"):
            features[index, genres.index(genre)] = 1
    lookup = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids))
    if ratings[:, 1].max() >= len(lookup):
        raise ValueError("Rating refers to an unknown movie")
    ratings[:, 1] = lookup[ratings[:, 1]]
    if np.any(ratings[:, 1] < 0) or np.any((ratings[:, 2] < 1) | (ratings[:, 2] > 5)):
        raise ValueError("Invalid rating or movie identifier")
    catalog = Catalog(ids, tuple(row[1] for row in movies), features, genres)
    return catalog, ratings


def split_users(ratings: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    users = np.unique(ratings[:, 0])
    if len(users) < 10:
        raise ValueError("At least ten users are needed for disjoint splits")
    shuffled = np.random.default_rng(seed).permutation(users)
    train_end, validation_end = int(0.7 * len(users)), int(0.85 * len(users))
    return {
        "train": np.sort(shuffled[:train_end]),
        "validation": np.sort(shuffled[train_end:validation_end]),
        "test": np.sort(shuffled[validation_end:]),
    }


def user_sequences(ratings: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, set[int]]]:
    order = np.lexsort((ratings[:, 1], ratings[:, 3], ratings[:, 0]))
    ordered = ratings[order]
    users, starts = np.unique(ordered[:, 0], return_index=True)
    sequences, observed = {}, {}
    for user, rows in zip(users, np.split(ordered, starts[1:]), strict=True):
        sequences[int(user)] = rows[rows[:, 2] >= 4, 1]
        observed[int(user)] = set(map(int, rows[:, 1]))
    return sequences, observed


def training_examples(
    sequences: dict[int, np.ndarray], users: np.ndarray, history_size: int = 20, per_user: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(history_size, per_user) < 1:
        raise ValueError("History size and examples per user must be positive")
    histories, targets, owners = [], [], []
    for user in users:
        items = sequences[int(user)]
        for position in range(max(1, len(items) - per_user), len(items)):
            prefix = items[max(0, position - history_size) : position] + 1
            padded = np.zeros(history_size, dtype=np.int64)
            padded[: len(prefix)] = prefix
            histories.append(padded)
            targets.append(int(items[position]))
            owners.append(int(user))
    if not targets:
        raise ValueError("No training users have two positive ratings")
    return np.stack(histories), np.array(targets), np.array(owners)
