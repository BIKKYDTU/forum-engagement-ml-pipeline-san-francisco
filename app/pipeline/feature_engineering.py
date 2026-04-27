"""Feature engineering for the Forum Thread Engagement Prediction Pipeline.

This module exposes :class:`ThreadFeatureExtractor`, which converts a single
thread record dictionary into the nine engineered features described in the
prompt. The extractor is used both directly by the test suite and by
:class:`pipeline.engagement_pipeline.EngagementPipeline` during training and
scoring.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict

FEATURE_NAMES = (
    "post_length",
    "has_question",
    "has_list",
    "has_media",
    "poster_activity",
    "hour_of_day",
    "day_of_week",
    "tag_count",
    "has_high_engagement_tag",
)

# Markdown list markers recognised by the ``has_list`` feature.
_LIST_MARKERS = ("- ", "* ", "1. ")

# URL whose path ends with one of these extensions indicates embedded media.
_MEDIA_URL_RE = re.compile(
    r"https?://\S+?\.(?:jpg|png|gif|mp4)\b",
    re.IGNORECASE,
)


class ThreadFeatureExtractor:
    """Transform a single thread record dict into nine engineered features.

    Parameters
    ----------
    high_engagement_tags:
        A set of tag strings classified as "high engagement" during training.
        When empty (the default), the ``has_high_engagement_tag`` feature is
        always ``0``. During training the pipeline computes this set from the
        historical DataFrame and re-instantiates the extractor with it.
    """

    def __init__(
        self,
        high_engagement_tags: "frozenset[str]" = frozenset(),
    ) -> None:
        self.high_engagement_tags: "frozenset[str]" = frozenset(high_engagement_tags)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_features(self, thread: Dict[str, Any]) -> Dict[str, Any]:
        """Return the nine engineered features for a single thread record.

        The returned dict contains exactly the keys listed in
        :data:`FEATURE_NAMES` with native Python integer/float values (never
        ``numpy`` scalars) so downstream assertions on types hold.
        """
        body = thread.get("body", "") or ""
        tags = thread.get("tags", []) or []

        return {
            "post_length": self._post_length(body),
            "has_question": self._has_question(body),
            "has_list": self._has_list(body),
            "has_media": self._has_media(body),
            "poster_activity": float(thread.get("poster_history_count", 0)),
            "hour_of_day": self._hour_of_day(thread.get("posted_at", "")),
            "day_of_week": self._day_of_week(thread.get("posted_at", "")),
            "tag_count": int(len(tags)),
            "has_high_engagement_tag": self._has_high_engagement_tag(tags),
        }

    # ------------------------------------------------------------------ #
    # Per-feature helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _post_length(body: str) -> int:
        return int(len(body.split()))

    @staticmethod
    def _has_question(body: str) -> int:
        return 1 if "?" in body else 0

    @staticmethod
    def _has_list(body: str) -> int:
        return 1 if any(marker in body for marker in _LIST_MARKERS) else 0

    @staticmethod
    def _has_media(body: str) -> int:
        if "![" in body:
            return 1
        return 1 if _MEDIA_URL_RE.search(body) else 0

    @staticmethod
    def _parse_timestamp(posted_at: str) -> datetime:
        # ``datetime.fromisoformat`` handles the standard ISO 8601 timestamps
        # described in the prompt (e.g. ``"2024-01-15T14:30:00"``).  A trailing
        # ``Z`` is tolerated for convenience.
        value = posted_at.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)

    @classmethod
    def _hour_of_day(cls, posted_at: str) -> int:
        return int(cls._parse_timestamp(posted_at).hour)

    @classmethod
    def _day_of_week(cls, posted_at: str) -> int:
        return int(cls._parse_timestamp(posted_at).weekday())

    def _has_high_engagement_tag(self, tags) -> int:
        if not self.high_engagement_tags:
            return 0
        return 1 if any(tag in self.high_engagement_tags for tag in tags) else 0
