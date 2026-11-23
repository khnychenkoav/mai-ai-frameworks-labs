"""Четыре режима оценки с одинаковым протоколом для каждой лабораторной."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, mean_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from labkit.data import ROOT, SEED, layout_features
from labkit.models import BootstrapForest, Cart, GradientTrees, LinearLeastSquares, Neighbors, SoftmaxRegression


VERSIONS = ("sklearn baseline", "sklearn tuned", "own baseline", "own tuned")
ALGORITHMS = {
    1: "KNN",
    2: "Логистическая / линейная регрессии",
    3: "Решающее дерево",
    4: "Случайный лес",
    5: "Градиентный бустинг",
}


@dataclass(frozen=True)
class Candidate:
    label: str
    preparation: str = "raw"
    parameters: dict | None = None


def _candidates(lab, classification):
    if lab == 1:
        if classification:
            return [
                Candidate("Бейзлайн"),
                Candidate("Стандартизация", "standard"),
                Candidate("Робастное масштабирование", "robust"),
                Candidate("k=3 и взвешивание соседей", "standard", {"n_neighbors": 3, "weights": "distance"}),
                Candidate("Манхэттенское расстояние и k=11", "standard", {"n_neighbors": 11, "p": 1}),
            ]
        return [
            Candidate("Бейзлайн"),
            Candidate("Стандартизация координат", "standard"),
            Candidate("Геометрия со стандартизацией", "geometry_standard"),
            Candidate("Геометрия, k=11 и взвешивание", "geometry_standard", {"n_neighbors": 11, "weights": "distance"}),
            Candidate("Координаты, k=11 и взвешивание", "raw", {"n_neighbors": 11, "weights": "distance"}),
        ]
    if lab == 2:
        if classification:
            return [
                Candidate("Бейзлайн"),
                Candidate("Взвешивание редких классов", "raw", {"class_weight": "balanced"}),
                Candidate("Стандартизация и C=0.3", "standard", {"C": 0.3}),
                Candidate("Квадратичные признаки и C=0.3", "polynomial", {"C": 0.3}),
                Candidate("Квадратичные признаки и веса классов", "polynomial", {"C": 0.3, "class_weight": "balanced"}),
            ]
        return [
            Candidate("Бейзлайн"),
            Candidate("Геометрия", "geometry"),
            Candidate("Геометрия и Ridge", "geometry_standard", {"alpha": 10.0}),
            Candidate("Квадратичные признаки координат и Ridge", "polynomial", {"alpha": 10.0}),
            Candidate("Квадратичные признаки геометрии и Ridge", "geometry_polynomial", {"alpha": 10.0}),
        ]
    if lab == 3:
        regularized = {"max_depth": 12, "min_samples_leaf": 3}
        if classification:
            return [
                Candidate("Бейзлайн"),
                Candidate("Глубина 12 и минимум 3 в листе", "raw", regularized),
                Candidate("Только веса классов", "raw", {"class_weight": "balanced"}),
                Candidate("Веса классов и ограничения дерева", "raw", {**regularized, "class_weight": "balanced"}),
                Candidate("Минимум 2 в листе", "raw", {"min_samples_leaf": 2}),
            ]
        return [
            Candidate("Бейзлайн"),
            Candidate("Глубина 12 и минимум 3 в листе", "raw", regularized),
            Candidate("Геометрия", "geometry"),
            Candidate("Геометрия и ограничения дерева", "geometry", regularized),
        ]
    if lab == 4:
        regularized = {"max_depth": 14, "min_samples_leaf": 3}
        if classification:
            return [
                Candidate("Бейзлайн"),
                Candidate("Взвешивание редких классов", "raw", {"class_weight": "balanced"}),
                Candidate("Глубина 14 и минимум 3 в листе", "raw", regularized),
                Candidate("64 дерева", "raw", {"n_estimators": 64}),
            ]
        return [
            Candidate("Бейзлайн"),
            Candidate("Геометрия", "geometry"),
            Candidate("64 дерева", "raw", {"n_estimators": 64}),
            Candidate("Геометрия и ограничения деревьев", "geometry", regularized),
            Candidate("Координаты и геометрия", "combined"),
        ]
    if lab == 5:
        deeper = {"max_depth": 3, "n_estimators": 48}
        if classification:
            return [
                Candidate("Бейзлайн"),
                Candidate("Глубина 3 и 48 итераций", "raw", deeper),
                Candidate("Глубина 4", "raw", {"max_depth": 4}),
                Candidate("Меньший шаг и 64 итерации", "raw", {"learning_rate": 0.05, "n_estimators": 64}),
            ]
        return [
            Candidate("Бейзлайн"),
            Candidate("Геометрия", "geometry"),
            Candidate("Глубина 3 и 48 итераций", "raw", deeper),
            Candidate("Геометрия и глубина 3", "geometry", deeper),
            Candidate("Координаты и геометрия, глубина 3", "combined", deeper),
        ]
    raise ValueError(f"Неизвестная лабораторная: {lab}")


def _combined_layout(X):
    return np.column_stack((X, layout_features(X)))


def _preprocessor(preparation):
    transforms = {
        "raw": [FunctionTransformer()],
        "standard": [StandardScaler()],
        "robust": [RobustScaler()],
        "geometry": [FunctionTransformer(layout_features)],
        "geometry_standard": [FunctionTransformer(layout_features), StandardScaler()],
        "combined": [FunctionTransformer(_combined_layout)],
        "polynomial": [StandardScaler(), PolynomialFeatures(include_bias=False), StandardScaler()],
        "geometry_polynomial": [
            FunctionTransformer(layout_features), StandardScaler(),
            PolynomialFeatures(include_bias=False), StandardScaler(),
        ],
    }
    return make_pipeline(*transforms[preparation])


def _model(lab, classification, candidate, own):
    parameters = candidate.parameters or {}
    if lab == 1:
        if own:
            return Neighbors(classification, **parameters)
        estimator = KNeighborsClassifier if classification else KNeighborsRegressor
        return estimator(**parameters)
    if lab == 2:
        if classification:
            arguments = {"max_iter": 1500, **parameters}
            return SoftmaxRegression(**arguments) if own else LogisticRegression(**arguments)
        if own:
            return LinearLeastSquares(**parameters)
        return Ridge(**parameters) if parameters else LinearRegression()
    if lab == 3:
        arguments = {"random_state": SEED, **parameters}
        if own:
            return Cart(classification, **arguments)
        estimator = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return estimator(**arguments)
    if lab == 4:
        arguments = {
            "n_estimators": 32, "max_features": "sqrt", "random_state": SEED, **parameters,
        }
        if own:
            return BootstrapForest(classification, **arguments)
        estimator = RandomForestClassifier if classification else RandomForestRegressor
        return estimator(n_jobs=1, **arguments)
    arguments = {
        "n_estimators": 32, "max_depth": 2, "min_samples_leaf": 5,
        "random_state": SEED, **parameters,
    }
    if own:
        return GradientTrees(classification, **arguments)
    estimator = GradientBoostingClassifier if classification else GradientBoostingRegressor
    return estimator(**arguments)


def primary_metric(task):
    return "macro_f1" if task.key == "classification" else "mae_mw"


def _primary_score(task, y, prediction):
    if task.key == "classification":
        return f1_score(y, prediction, labels=np.unique(task.y_train), average="macro", zero_division=0)
    return mean_absolute_error(y, prediction)


def _all_scores(task, prediction):
    y = task.y_test
    if task.key == "classification":
        return {
            "macro_f1": _primary_score(task, y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "accuracy": accuracy_score(y, prediction),
        }
    return {
        "mae_mw": _primary_score(task, y, prediction),
        "rmse_mw": root_mean_squared_error(y, prediction),
        "r2": r2_score(y, prediction),
    }


def _fit(lab, task, candidate, own, X, y):
    transform = _preprocessor(candidate.preparation)
    model = _model(lab, task.key == "classification", candidate, own)
    model.fit(transform.fit_transform(X), y)

    def predict(new_X):
        return model.predict(transform.transform(new_X))

    return predict


def _search(lab, task):
    candidates = _candidates(lab, task.key == "classification")
    rows = []
    for candidate in candidates:
        scores = []
        for train, valid in task.cv:
            predict = _fit(lab, task, candidate, False, task.X_train[train], task.y_train[train])
            prediction = predict(task.X_train[valid])
            scores.append(_primary_score(task, task.y_train[valid], prediction))
        rows.append({
            "гипотеза": candidate.label, "признаки": candidate.preparation,
            "настройки": json.dumps(candidate.parameters or {}, ensure_ascii=False),
            "CV среднее": np.mean(scores), "CV σ": np.std(scores, ddof=1),
            **{f"фолд {i}": score for i, score in enumerate(scores, 1)},
        })
    table = pd.DataFrame(rows)
    means = table["CV среднее"]
    best = means.idxmax() if task.key == "classification" else means.idxmin()
    table["выбран"] = table.index == best
    return candidates[0], candidates[best], table


def select_candidates(lab, tasks):
    """Перебирает гипотезы только на фолдах обучающей выборки."""
    with threadpool_limits(limits=1):
        return {task.key: _search(lab, task) for task in tasks}


def _evaluate(lab, task, baseline, tuned):
    rows, predictions = [], {}
    classification = task.key == "classification"
    dummy = DummyClassifier(strategy="most_frequent") if classification else DummyRegressor(strategy="median")
    dummy.fit(task.X_train, task.y_train)
    rows.append({
        "lab": lab, "task": task.key, "algorithm": ALGORITHMS[lab], "version": "dummy",
        "candidate": "Мода" if classification else "Медиана", "fit_s": np.nan, "predict_s": np.nan,
        **_all_scores(task, dummy.predict(task.X_test)),
    })
    for version, candidate, own in zip(VERSIONS, (baseline, tuned, baseline, tuned), (False, False, True, True)):
        start = perf_counter()
        predict = _fit(lab, task, candidate, own, task.X_train, task.y_train)
        fit_time = perf_counter() - start
        start = perf_counter()
        prediction = predict(task.X_test)
        predict_time = perf_counter() - start
        predictions[version] = prediction
        rows.append({
            "lab": lab, "task": task.key, "algorithm": ALGORITHMS[lab], "version": version,
            "candidate": candidate.label, "fit_s": fit_time, "predict_s": predict_time,
            **_all_scores(task, prediction),
        })
    return pd.DataFrame(rows), predictions


def evaluate_holdout(lab, tasks, selections):
    """Обучает финальные конфигурации на всём train и оценивает holdout."""
    with threadpool_limits(limits=1):
        return {task.key: _evaluate(lab, task, *selections[task.key][:2]) for task in tasks}


def describe_data(tasks):
    return pd.DataFrame([
        {
            "задача": task.title, "строк в исходных файлах": task.source_rows,
            "train": len(task.y_train), "test": len(task.y_test),
            "исходные признаки": len(task.feature_names), "разбиение": task.description,
        }
        for task in tasks
    ])


def plot_training(tasks):
    classification, regression = tasks
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    labels, counts = np.unique(classification.y_train, return_counts=True)
    axes[0].bar(labels, counts, color="#286780")
    axes[0].set(xlabel="Код переписчика", ylabel="Фрагменты", title="Распределение классов в train")
    axes[1].hist(regression.y_train, bins=30, color="#51826d")
    axes[1].set(xlabel="Суммарная мощность, МВт", ylabel="Конфигурации", title="Распределение мощности в train")
    plt.show()


def plot_predictions(tasks, results):
    classification, regression = tasks
    labels = np.unique(classification.y_train)
    class_prediction = results[classification.key][1]["sklearn tuned"]
    regression_prediction = results[regression.key][1]["sklearn tuned"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    matrix = confusion_matrix(classification.y_test, class_prediction, labels=labels, normalize="true")
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axes[0], include_values=False, cmap="Blues", colorbar=True,
    )
    axes[0].set(title="Avila · доли в строках", xlabel="Предсказанный класс", ylabel="Истинный класс")
    observed = regression.y_test
    axes[1].scatter(observed, regression_prediction, s=8, alpha=0.30, color="#286780")
    limits = [min(observed.min(), regression_prediction.min()), max(observed.max(), regression_prediction.max())]
    axes[1].plot(limits, limits, color="#b34d37", linestyle="--", linewidth=1)
    axes[1].set(title="Tasmania · прогноз против значения", xlabel="Мощность по симулятору, МВт",
                ylabel="Предсказанная мощность, МВт")
    plt.show()


def save_scores(lab, results):
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    path = directory / f"lab_{lab:02d}.csv"
    columns_to_skip = ["fit_s", "predict_s"]
    scores = pd.concat([table for table, _ in results.values()], ignore_index=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=directory, suffix=".csv.tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            scores.drop(columns=columns_to_skip).to_csv(output, index=False, float_format="%.10g")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path.relative_to(ROOT)


def comparison_text(tasks, results, selections):
    lines = []
    for task in tasks:
        metric = primary_metric(task)
        values = results[task.key][0].set_index("version")[metric]
        base, tuned, own_base, own_tuned = values.loc[list(VERSIONS)]
        validation = selections[task.key][2]
        selected = validation.loc[validation["выбран"]].iloc[0]
        cv_base = validation.iloc[0]["CV среднее"]
        if task.key == "classification":
            name, unit, direction = "macro-F1", "", 1
        else:
            name, unit, direction = "MAE", " МВт", -1
        heldout_gain = (tuned - base) * direction
        verdict = "улучшился" if heldout_gain > 1e-8 else "ухудшился" if heldout_gain < -1e-8 else "совпал"
        lines.append(
            f"**{task.title}.** По CV выбран вариант «{selected['гипотеза']}» "
            f"({name} {selected['CV среднее']:.4f}{unit}, бейзлайн {cv_base:.4f}{unit}). "
            f"На holdout {name} sklearn {verdict} относительно бейзлайна: "
            f"{base:.4f} → {tuned:.4f}{unit}. Собственная реализация: "
            f"{own_base:.4f} → {own_tuned:.4f}{unit}. "
            f"Разница собственной модели и sklearn {name}: "
            f"{own_base - base:+.4f}{unit} в бейзлайне, "
            f"{own_tuned - tuned:+.4f}{unit} после настройки."
        )
    return "\n\n".join(lines)
