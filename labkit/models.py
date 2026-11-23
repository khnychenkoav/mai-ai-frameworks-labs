"""Реализации оценивателей без вызовов моделей scikit-learn."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax


def _class_weights(labels, n_classes, mode):
    if mode == "balanced":
        counts = np.bincount(labels, minlength=n_classes)
        return len(labels) / (n_classes * counts[labels])
    return np.ones(len(labels))


class Neighbors:
    """Brute-force KNN с пакетным вычислением расстояний."""

    def __init__(self, classification, n_neighbors=5, p=2, weights="uniform"):
        self.classification = classification
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        if not 1 <= self.n_neighbors <= len(self.X_):
            raise ValueError("n_neighbors должен лежать в диапазоне от 1 до размера train")
        if self.classification:
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
        else:
            self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        results = []
        k = self.n_neighbors
        metrics = {1: "cityblock", 2: "euclidean"}
        metric = metrics.get(self.p, "minkowski")
        parameters = {"p": self.p} if metric == "minkowski" else {}
        for start in range(0, len(X), 128):
            distance = cdist(X[start:start + 128], self.X_, metric=metric, **parameters)
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            distances = np.take_along_axis(distance, nearest, axis=1)
            boundary = distances.max(axis=1)
            # При равенстве на границе k берём соседей в порядке строк train.
            total_tied = (distance == boundary[:, None]).sum(axis=1)
            selected_tied = (distances == boundary[:, None]).sum(axis=1)
            for row in np.flatnonzero(total_tied > selected_tied):
                nearest[row] = np.argsort(distance[row], kind="stable")[:k]
            distances = np.take_along_axis(distance, nearest, axis=1)
            weights = np.ones_like(distances)
            if self.weights == "distance":
                zero = distances == 0
                np.divide(1, distances, out=weights, where=~zero)
                weights = np.where(zero.any(axis=1, keepdims=True), zero, weights)
            labels = self.y_[nearest]
            if self.classification:
                votes = np.zeros((len(nearest), len(self.classes_)))
                np.add.at(votes, (np.arange(len(nearest))[:, None], labels), weights)
                results.append(self.classes_[votes.argmax(axis=1)])
            else:
                results.append(np.sum(labels * weights, axis=1) / weights.sum(axis=1))
        if not results:
            return self.y_[:0] if not self.classification else self.classes_[:0]
        return np.concatenate(results)


class LinearLeastSquares:
    """OLS и L2-регуляризация через расширенную задачу МНК."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)
        design, target = X - self.x_mean_, y - self.y_mean_
        if self.alpha > 0:
            dimension = X.shape[1]
            design = np.vstack((design, np.sqrt(self.alpha) * np.eye(dimension)))
            target = np.concatenate((target, np.zeros(dimension)))
        self.coef_ = np.linalg.lstsq(design, target, rcond=None)[0]
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class SoftmaxRegression:
    """Мультиномиальная cross-entropy, аналитический градиент и L-BFGS."""

    def __init__(self, C=1.0, class_weight=None, max_iter=500):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_, labels = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Для softmax нужны как минимум два класса")
        design = np.column_stack((X, np.ones(len(X))))
        weights = _class_weights(labels, n_classes, self.class_weight)
        rows = np.arange(len(X))
        s = weights.sum()
        regularization = 1 / (self.C * s)

        def objective(flat):
            coefficients = flat.reshape(design.shape[1], n_classes)
            logits = design @ coefficients
            normalizers = logsumexp(logits, axis=1)
            loss = weights @ (normalizers - logits[rows, labels]) / s
            loss += 0.5 * regularization * np.square(coefficients[:-1]).sum()
            errors = np.exp(logits - normalizers[:, None])
            errors[rows, labels] -= 1
            gradient = design.T @ (errors * (weights / s)[:, None])
            gradient[:-1] += regularization * coefficients[:-1]
            return loss, gradient.ravel()

        fit = minimize(
            objective, np.zeros(design.shape[1] * n_classes), jac=True,
            method="L-BFGS-B", options={"maxiter": self.max_iter, "gtol": 1e-6, "ftol": 1e-11},
        )
        self.coef_ = fit.x.reshape(design.shape[1], n_classes)
        self.n_iter_ = fit.nit
        self.converged_ = fit.success
        if not fit.success:
            raise RuntimeError(f"L-BFGS не сошёлся: {fit.message}")
        return self

    def predict_proba(self, X):
        return softmax(X @ self.coef_[:-1] + self.coef_[-1], axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


@njit(cache=True)
def _best_split(X, y, weight, features, min_leaf, n_classes):
    """Полный перебор вещественных порогов с префиксными статистиками."""
    n = len(y)
    best_score = np.inf
    best_feature = -1
    best_threshold = 0.0
    weight_total = weight.sum()
    total_counts = np.zeros(n_classes)
    if n_classes:
        for i in range(n):
            total_counts[int(y[i])] += weight[i]
    else:
        weighted_y = y * weight
        total_sum = weighted_y.sum()
        total_square = (weighted_y * y).sum()

    for feature in features:
        values = X[:, feature]
        order = np.argsort(values)
        left_weight = 0.0
        left_counts = np.zeros(n_classes)
        left_sum = 0.0
        left_square = 0.0
        for idx in range(n - min_leaf):
            row = order[idx]
            left_weight += weight[row]
            if n_classes:
                left_counts[int(y[row])] += weight[row]
            else:
                left_sum += weight[row] * y[row]
                left_square += weight[row] * y[row] ** 2
            if idx + 1 < min_leaf or values[row] == values[order[idx + 1]]:
                continue
            right_weight = weight_total - left_weight
            if left_weight <= 0 or right_weight <= 0:
                continue
            if n_classes:
                score = weight_total
                for label in range(n_classes):
                    right_count = total_counts[label] - left_counts[label]
                    score -= left_counts[label] ** 2 / left_weight
                    score -= right_count ** 2 / right_weight
            else:
                right_sum = total_sum - left_sum
                right_square = total_square - left_square
                score = left_square - left_sum ** 2 / left_weight
                score += right_square - right_sum ** 2 / right_weight
            if score < best_score - 1e-12:
                best_score = score
                best_feature = feature
                next_value = values[order[idx + 1]]
                best_threshold = values[row] + (next_value - values[row]) / 2
                if best_threshold >= next_value:
                    best_threshold = values[row]
    return best_feature, best_threshold, best_score


@njit(cache=True)
def _apply_tree(X, features, thresholds, left, right):
    leaves = np.empty(len(X), dtype=np.int64)
    for row in range(len(X)):
        node = 0
        while features[node] >= 0:
            if X[row, features[node]] <= thresholds[node]:
                node = left[node]
            else:
                node = right[node]
        leaves[row] = node
    return leaves


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1


class Cart:
    """Бинарное дерево CART с критерием Gini или SSE."""

    def __init__(
        self, classification, max_depth=None, min_samples_leaf=1, max_features=None,
        class_weight=None, random_state=0,
    ):
        self.classification = classification
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        if self.classification:
            self.classes_, labels = np.unique(y, return_inverse=True)
            self._fit(X, labels, len(self.classes_), _class_weights(
                labels, len(self.classes_), self.class_weight
            ))
        else:
            self._fit(X, np.asarray(y, dtype=float), 0, np.ones(len(y)))
        return self

    def _fit(self, X, target, n_classes, weights):
        X = np.asarray(X, dtype=float)
        target = np.asarray(target, dtype=float)
        n_features = X.shape[1]
        feature_count = n_features if self.max_features is None else max(1, int(np.sqrt(n_features)))
        rng = np.random.default_rng(self.random_state)
        nodes = []
        values = []
        pending = [(np.arange(len(X)), 0, -1, False)]
        while pending:
            indices, depth, parent, is_left = pending.pop()
            node_id = len(nodes)
            node = _Node()
            nodes.append(node)
            if parent >= 0:
                if is_left:
                    nodes[parent].left = node_id
                else:
                    nodes[parent].right = node_id

            y_node, w_node = target[indices], weights[indices]
            weight_total = w_node.sum()
            if n_classes:
                prediction = np.bincount(
                    y_node.astype(int), weights=w_node, minlength=n_classes,
                ) / weight_total
                impurity = weight_total * (1 - prediction @ prediction)
            else:
                prediction = np.average(y_node, weights=w_node)
                impurity = w_node @ np.square(y_node - prediction)
            values.append(prediction)

            if (
                len(indices) < 2 * self.min_samples_leaf or impurity <= 1e-12
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                continue

            X_node = X[indices]
            features = rng.permutation(n_features)
            feature, threshold, score = _best_split(
                X_node, y_node, w_node, features[:feature_count], self.min_samples_leaf, n_classes
            )
            # Если подмножество признаков константно, ищем допустимый разрез дальше.
            if feature < 0 and feature_count < n_features:
                feature, threshold, score = _best_split(
                    X_node, y_node, w_node, features[feature_count:], self.min_samples_leaf, n_classes
                )
            if feature < 0 or impurity - score <= 1e-12:
                continue
            mask = X_node[:, feature] <= threshold
            node.feature, node.threshold = feature, threshold
            pending.append((indices[~mask], depth + 1, node_id, False))
            pending.append((indices[mask], depth + 1, node_id, True))

        self.features_ = np.array([node.feature for node in nodes])
        self.thresholds_ = np.array([node.threshold for node in nodes])
        self.left_ = np.array([node.left for node in nodes])
        self.right_ = np.array([node.right for node in nodes])
        self.values_ = np.asarray(values)
        return self

    def apply(self, X):
        return _apply_tree(np.asarray(X), self.features_, self.thresholds_, self.left_, self.right_)

    def _leaf_values(self, X):
        return self.values_[self.apply(X)]

    def predict(self, X):
        values = self._leaf_values(X)
        return self.classes_[values.argmax(axis=1)] if self.classification else values


class BootstrapForest:
    """Bootstrap объектов, случайные признаки в узлах, агрегация деревьев."""

    def __init__(
        self, classification, n_estimators=32, max_depth=None, min_samples_leaf=1,
        max_features=None, class_weight=None, random_state=0,
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_classes = 0
        weights = np.ones(len(y))
        if self.classification:
            self.classes_, y = np.unique(y, return_inverse=True)
            n_classes = len(self.classes_)
            weights = _class_weights(y, n_classes, self.class_weight)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            sample = rng.integers(len(X), size=len(X))
            tree = Cart(
                self.classification, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                random_state=int(rng.integers(2**31)),
            )
            tree._fit(X[sample], y[sample], n_classes, weights[sample])
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        predictions = np.mean([tree._leaf_values(X) for tree in self.estimators_], axis=0)
        return self.classes_[predictions.argmax(axis=1)] if self.classification else predictions


class GradientTrees:
    """Бустинг CART для квадратичной ошибки и мультиномиальной log-loss."""

    def __init__(
        self, classification, n_estimators=32, learning_rate=0.1, max_depth=2,
        min_samples_leaf=3, random_state=0,
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _tree(self, rng):
        return Cart(
            False, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            random_state=int(rng.integers(2**31)),
        )

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        if not self.classification:
            self.initial_ = float(np.mean(y))
            prediction = np.full(len(X), self.initial_)
            for _ in range(self.n_estimators):
                tree = self._tree(rng).fit(X, y - prediction)
                prediction += self.learning_rate * tree.predict(X)
                self.estimators_.append((tree,))
            return self

        self.classes_, labels = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        target = np.eye(n_classes)[labels]
        self.initial_ = np.log(target.mean(axis=0))
        logits = np.tile(self.initial_, (len(X), 1))
        correction = (n_classes - 1) / n_classes
        for _ in range(self.n_estimators):
            probabilities = softmax(logits, axis=1)
            residuals = target - probabilities
            hessians = probabilities * (1 - probabilities)
            stage = []
            for label in range(n_classes):
                tree = self._tree(rng).fit(X, residuals[:, label])
                leaves = tree.apply(X)
                n_nodes = len(tree.values_)
                num = np.bincount(leaves, weights=residuals[:, label], minlength=n_nodes)
                den = np.bincount(leaves, weights=hessians[:, label], minlength=n_nodes)
                step = np.divide(correction * num, den, out=np.zeros(n_nodes), where=den > 1e-12)
                tree.values_ = step
                logits[:, label] += self.learning_rate * step[leaves]
                stage.append(tree)
            self.estimators_.append(tuple(stage))
        return self

    def predict(self, X):
        if not self.classification:
            result = np.full(len(X), self.initial_)
            for (tree,) in self.estimators_:
                result += self.learning_rate * tree.predict(X)
            return result
        logits = np.tile(self.initial_, (len(X), 1))
        for stage in self.estimators_:
            logits += self.learning_rate * np.column_stack([tree.predict(X) for tree in stage])
        return self.classes_[logits.argmax(axis=1)]
