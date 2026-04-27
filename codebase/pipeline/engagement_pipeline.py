"""End-to-end engagement prediction pipeline.

This module implements :func:`create_engagement_labels` and
:class:`EngagementPipeline`, the two public interfaces (in addition to
:class:`pipeline.feature_engineering.ThreadFeatureExtractor`) that together
satisfy the Forum Thread Engagement Prediction Pipeline prompt.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline.feature_engineering import FEATURE_NAMES, ThreadFeatureExtractor


# ---------------------------------------------------------------------------- #
# Engagement label generation
# ---------------------------------------------------------------------------- #
def create_engagement_labels(df: pd.DataFrame) -> pd.Series:
    """Return a binary engagement label Series using the OR-of-medians rule.

    A row is labeled ``1`` when its ``comment_count_48h`` is **strictly**
    greater than the median ``comment_count_48h`` across ``df``, OR its
    ``vote_count_48h`` is **strictly** greater than the median
    ``vote_count_48h`` across ``df``. Otherwise it is labeled ``0``.

    The returned Series has the same length and index as ``df``, integer
    dtype, and values drawn exclusively from ``{0, 1}``.
    """
    comment_median = df["comment_count_48h"].median()
    vote_median = df["vote_count_48h"].median()

    labels = (
        (df["comment_count_48h"] > comment_median)
        | (df["vote_count_48h"] > vote_median)
    ).astype(int)

    labels.index = df.index
    return labels


# ---------------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------------- #
class EngagementPipeline:
    """Compose a :class:`ThreadFeatureExtractor` with a Random Forest classifier.

    Parameters
    ----------
    n_estimators:
        Number of trees in the Random Forest. Defaults to ``100``.
    random_state:
        Seed used by the classifier for reproducibility. Defaults to ``42``.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.n_estimators: int = int(n_estimators)
        self.random_state: int = int(random_state)

        self.feature_extractor: ThreadFeatureExtractor = ThreadFeatureExtractor()
        self.classifier: Optional[RandomForestClassifier] = None
        self.training_mean: Optional[np.ndarray] = None
        self.training_std: Optional[np.ndarray] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, historical_data: pd.DataFrame) -> None:
        """Fit the classifier on ``historical_data`` and prepare scoring state.

        Steps performed:

        1. Derive engagement labels using :func:`create_engagement_labels`.
        2. Compute the set of "high-engagement" tags — those whose per-tag
           median ``comment_count_48h`` is strictly greater than the global
           median ``comment_count_48h`` of the training set.
        3. Instantiate a :class:`ThreadFeatureExtractor` configured with that
           tag set and build the feature matrix for every training row.
        4. Store the column-wise training mean and standard deviation of the
           feature matrix for later per-instance contribution scoring.
        5. Fit a :class:`RandomForestClassifier` on the features and labels.

        Returns ``None``.
        """
        labels = create_engagement_labels(historical_data)

        high_engagement_tags = self._compute_high_engagement_tags(historical_data)
        self.feature_extractor = ThreadFeatureExtractor(
            high_engagement_tags=high_engagement_tags
        )

        feature_matrix = self._build_feature_matrix(
            historical_data.to_dict(orient="records")
        )

        self.training_mean = feature_matrix.mean(axis=0)
        self.training_std = feature_matrix.std(axis=0)

        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.classifier.fit(feature_matrix, labels.to_numpy())

        self._is_trained = True
        return None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_threads(self, new_threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of new threads and return probability + top factors.

        Each returned dict contains:

        - ``score``: positive-class probability from the classifier's
          ``predict_proba``, as a native Python ``float`` in ``[0.0, 1.0]``.
        - ``top_factors``: the three feature names with the largest
          per-instance contribution scores for that thread, sorted in
          descending order. The per-instance contribution for feature ``i``
          is ``feature_importances_[i] * |z_score[i]|`` where
          ``z_score[i] = (x[i] - training_mean[i]) / training_std[i]``
          (with ``training_std[i]`` clamped to ``1.0`` when the column's
          training standard deviation is zero).

        Raises
        ------
        RuntimeError
            If called before :meth:`train`.
        """
        if not self._is_trained or self.classifier is None:
            raise RuntimeError(
                "EngagementPipeline.score_threads() called before train(); "
                "the pipeline must be trained on historical data first."
            )

        if len(new_threads) == 0:
            return []

        feature_matrix = self._build_feature_matrix(new_threads)
        probabilities = self.classifier.predict_proba(feature_matrix)
        positive_scores = self._positive_class_scores(probabilities)

        importances = np.asarray(self.classifier.feature_importances_, dtype=float)
        safe_std = np.where(self.training_std == 0, 1.0, self.training_std)

        results: List[Dict[str, Any]] = []
        for row_idx, _thread in enumerate(new_threads):
            feature_row = feature_matrix[row_idx]
            z_scores = (feature_row - self.training_mean) / safe_std
            contributions = importances * np.abs(z_scores)

            top_indices = np.argsort(-contributions, kind="stable")[:3]
            top_factors = [FEATURE_NAMES[i] for i in top_indices]

            results.append(
                {
                    "score": float(positive_scores[row_idx]),
                    "top_factors": top_factors,
                }
            )

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_high_engagement_tags(df: pd.DataFrame) -> "frozenset[str]":
        """Return the set of tags whose per-tag median comment count beats the global median."""
        global_median = df["comment_count_48h"].median()

        tag_to_counts: Dict[str, List[float]] = {}
        for tags, comment_count in zip(df["tags"].tolist(), df["comment_count_48h"].tolist()):
            if not tags:
                continue
            for tag in tags:
                tag_to_counts.setdefault(tag, []).append(float(comment_count))

        high_engagement = {
            tag
            for tag, counts in tag_to_counts.items()
            if float(np.median(counts)) > float(global_median)
        }
        return frozenset(high_engagement)

    def _build_feature_matrix(self, records: Iterable[Dict[str, Any]]) -> np.ndarray:
        """Extract features for each record and stack them into a 2D array."""
        rows = []
        for record in records:
            features = self.feature_extractor.extract_features(record)
            rows.append([float(features[name]) for name in FEATURE_NAMES])
        return np.asarray(rows, dtype=float)

    def _positive_class_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Return the probability of the positive class (label ``1``).

        When the training set contains only a single class the classifier's
        ``predict_proba`` output has a single column; in that case we fall
        back to ``1.0`` or ``0.0`` depending on which class was observed.
        This keeps ``score_threads`` robust even on degenerate training data.
        """
        classes = getattr(self.classifier, "classes_", np.array([0, 1]))
        if probabilities.shape[1] == 1:
            only_class = classes[0]
            fill = 1.0 if int(only_class) == 1 else 0.0
            return np.full(probabilities.shape[0], fill, dtype=float)

        positive_col = int(np.where(classes == 1)[0][0])
        return probabilities[:, positive_col].astype(float)
