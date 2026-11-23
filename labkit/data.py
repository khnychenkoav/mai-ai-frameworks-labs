"""Загрузка данных и разбиение выборок."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
SEED = 40122
AVILA_COLUMNS = (
    "intercolumnar_distance", "upper_margin", "lower_margin", "exploitation",
    "row_number", "modular_ratio", "interlinear_spacing", "weight", "peak_number",
    "modular_ratio_over_spacing",
)
LAYOUT_COLUMNS = tuple(f"{axis}{i}" for axis in ("X", "Y") for i in range(1, 17))
SOURCES = {
    "avila": (
        "https://archive.ics.uci.edu/static/public/459/avila.zip",
        "6a7e552eb45f8766a4207a6407fcff98d25ac5482759f2ca84a871dd9f5ca202",
    ),
    "wave": (
        "https://archive.ics.uci.edu/static/public/494/wave+energy+converters.zip",
        "b2c50eff48c11da423703221dbc5724c077fd5767d2e11d4291a2bfae0f61067",
    ),
}


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    feature_names: tuple
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    cv: tuple
    source_rows: int
    description: str
    audit: dict


def _checksum(path):
    with path.open("rb") as source:
        digest = sha256()
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(key):
    url, expected = SOURCES[key]
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    path = directory / f"{key}.zip"
    if path.exists() and _checksum(path) == expected:
        return ZipFile(path)
    request = Request(url, headers={"User-Agent": "mai-ai-frameworks-labs/1.0"})
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = Path(file.name)
            with urlopen(request, timeout=90) as response:
                for block in iter(lambda: response.read(1024**2), b""):
                    file.write(block)
        if _checksum(temporary) != expected:
            raise ValueError(f"Изменилось содержимое архива UCI: {url}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return ZipFile(path)


@lru_cache(maxsize=None)
def load_classification():
    with _archive("avila") as archive:
        names = [*AVILA_COLUMNS, "copyist"]
        with archive.open("avila/avila-tr.txt") as source:
            train = pd.read_csv(source, header=None, names=names)
        with archive.open("avila/avila-ts.txt") as source:
            test = pd.read_csv(source, header=None, names=names)

    X_train = train.iloc[:, :-1].to_numpy()
    X_test = test.iloc[:, :-1].to_numpy()
    y_train, y_test = train.copyist.to_numpy(), test.copyist.to_numpy()
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert set(y_train) == set(y_test)
    overlaps = len(set(map(tuple, X_train)) & set(map(tuple, X_test)))
    # В исходных файлах строки идут по классам.
    cv = tuple(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train, y_train))
    return Task(
        "classification", "Avila", AVILA_COLUMNS, X_train, X_test,
        y_train, y_test, cv, len(train) + len(test),
        "Разбиение UCI, 3 стратифицированных фолда на train.",
        {"Точных совпадений векторов признаков между train/test": overlaps},
    )


def _layout_keys(x, resolution=None):
    coordinates = x.reshape(-1, 2, 16).transpose(0, 2, 1)
    if resolution is not None:
        coordinates = np.rint(coordinates / resolution).astype(np.int16)
    order = np.lexsort((coordinates[:, :, 1], coordinates[:, :, 0]), axis=1)
    canonical = np.take_along_axis(coordinates, order[:, :, None], axis=1)
    return canonical.reshape(-1, 32)


@lru_cache(maxsize=None)
def load_regression():
    with _archive("wave") as archive, archive.open("WECs_DataSet/Tasmania_Data.csv") as source:
        # Читаем 32 координаты и столбец Powerall.
        columns = [*range(32), 48]
        frame = pd.read_csv(source, header=None, usecols=columns, dtype=np.float64)
    source_rows = len(frame)
    finite = frame.loc[np.isfinite(frame.to_numpy()).all(axis=1)].to_numpy()
    coordinates, power = finite[:, :32], finite[:, 32]
    valid = ((coordinates >= 0) & (coordinates <= 566)).all(axis=1) & (power > 0)
    X = coordinates[valid]
    y = power[valid] / 1_000_000  # Перевод мощности из Вт в МВт.

    # Удаляем одинаковые раскладки с учетом перестановок.
    _, distinct = np.unique(_layout_keys(X), axis=0, return_index=True)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(np.sort(distinct), size=min(7200, len(distinct)), replace=False)
    X, y = X[sample], y[sample]
    _, groups = np.unique(_layout_keys(X, resolution=10), axis=0, return_inverse=True)
    train, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    ).split(X, y, groups))
    assert not set(groups[train]) & set(groups[test])
    cv = tuple(GroupKFold(n_splits=3).split(X[train], y[train], groups[train]))
    return Task(
        "regression", "Tasmania", LAYOUT_COLUMNS,
        X[train], X[test], y[train], y[test], cv, source_rows,
        "20% групп на тесте, 3 групповых фолда на train.",
        {
            "Строк после проверки диапазонов и пропусков": int(valid.sum()),
            "Точных уникальных конфигураций": len(distinct),
            "Строк в воспроизводимой подвыборке": len(sample),
            "Групп в подвыборке (округление координат до 10 м)": len(np.unique(groups)),
        },
    )


def load_tasks():
    return load_classification(), load_regression()


def layout_features(X):
    """Геометрические признаки раскладки без учета нумерации."""
    layouts = X.reshape(-1, 2, 16).transpose(0, 2, 1)
    centers = layouts.mean(axis=1)
    spreads = layouts.std(axis=1)
    spans = np.ptp(layouts, axis=1)
    distances = np.stack([pdist(layout) for layout in layouts])
    quantiles = np.quantile(distances, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    edges = np.minimum(layouts, 566 - layouts).min(axis=2)
    return np.column_stack((
        centers, spreads, spans, distances.min(axis=1), distances.mean(axis=1),
        distances.std(axis=1), distances.max(axis=1), quantiles,
        edges.mean(axis=1), edges.min(axis=1),
    ))
