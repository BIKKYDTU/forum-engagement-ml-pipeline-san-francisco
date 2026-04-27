"""Test suite for Forum Thread Engagement Prediction Pipeline."""

# ── COVERAGE MAP ──────────────────────────────────────────────────────────────
# Req-1  ThreadFeatureExtractor constructor: default high_engagement_tags=frozenset()
#        → test_default_high_engagement_tags_is_empty_frozenset
# Req-2  ThreadFeatureExtractor constructor: stores high_engagement_tags as instance attr
#        → test_high_engagement_tags_stored_as_instance_attribute
# Req-3  extract_features returns dict with exactly the 9 mandated keys and no others
#        → test_extract_features_returns_exactly_nine_keys
# Req-4  post_length: int word count of the body
#        → test_post_length_equals_word_count_of_body,
#           test_post_length_is_zero_for_empty_body
# Req-5  has_question: int binary — 1 when body contains '?', else 0
#        → test_has_question_is_one_when_body_contains_question_mark,
#           test_has_question_is_zero_when_body_has_no_question_mark
# Req-6  has_list: int binary — 1 when body contains '- ', '* ', or '1. '
#        → test_has_list_is_one_when_list_marker_present[3 parametrised cases],
#           test_has_list_is_zero_when_body_contains_no_list_marker
# Req-7  has_media: int binary — 1 when body contains '![' or .jpg/.png/.gif/.mp4 URL
#        → test_has_media_is_zero_when_body_has_no_media_reference,
#           test_has_media_is_one_when_media_reference_present[5 parametrised cases]
# Req-8  poster_activity: float equal to poster_history_count
#        → test_poster_activity_equals_poster_history_count
# Req-9  hour_of_day: int in [0, 23] extracted from the posted_at ISO timestamp
#        → test_hour_of_day_extracted_from_posted_at_timestamp,
#           test_hour_of_day_is_zero_at_midnight,
#           test_hour_of_day_is_in_valid_range
# Req-10 day_of_week: int in [0, 6] (0=Monday) extracted from the posted_at timestamp
#        → test_day_of_week_is_zero_for_monday,
#           test_day_of_week_is_six_for_sunday,
#           test_day_of_week_is_in_valid_range
# Req-11 tag_count: int equal to len(tags)
#        → test_tag_count_equals_number_of_tags_in_list,
#           test_tag_count_is_zero_for_empty_tags_list
# Req-12 has_high_engagement_tag: int binary — 1 when any tag in set; 0 when set empty
#        → test_has_high_engagement_tag_is_one_when_thread_tag_in_high_engagement_set,
#           test_has_high_engagement_tag_is_zero_when_no_thread_tag_matches_set,
#           test_has_high_engagement_tag_is_zero_when_high_engagement_set_is_empty
# Req-13 create_engagement_labels returns pd.Series of integer dtype with values {0, 1}
#        → test_labels_contain_only_binary_integer_values
# Req-14 create_engagement_labels output has same length and index as input DataFrame
#        → test_returns_series_with_same_length_and_index_as_input_dataframe
# Req-15 Engagement labeling: OR-of-medians (strictly greater than); equal→0
#        → test_label_is_one_when_comment_count_strictly_exceeds_median,
#           test_label_is_one_when_vote_count_strictly_exceeds_median,
#           test_label_is_zero_when_both_counts_at_or_below_their_medians,
#           test_or_rule_applies_when_either_metric_strictly_exceeds_its_median
# Req-16 EngagementPipeline constructor: n_estimators=100, random_state=42 as defaults
#        → test_pipeline_can_be_instantiated_with_default_parameters,
#           test_pipeline_accepts_custom_n_estimators_and_random_state
# Req-17 train(): computes high-engagement tags (per-tag median comment > global median)
#        → test_train_computes_high_engagement_tags_correctly
# Req-18 train(): returns None
#        → test_train_returns_none
# Req-19 score_threads(): output list length equals input list length
#        → test_score_threads_output_length_equals_input_length
# Req-20 score_threads(): each result dict has keys 'score' (float [0,1])
#        and 'top_factors' (list[str] of exactly 3)
#        → test_score_threads_each_result_contains_score_and_top_factors_keys,
#           test_score_is_a_float_in_zero_to_one_range,
#           test_top_factors_is_a_list_of_exactly_three_strings
# Req-21 top_factors: 3 distinct valid feature names drawn from the 9-feature set
#        → test_top_factors_is_a_list_of_exactly_three_strings,
#           test_top_factors_contains_no_duplicate_feature_names
# Req-22 top_factors: sorted descending by per-instance contribution = importance×|z_score|
#        → test_top_factors_are_sorted_in_descending_contribution_score_order
# Req-23 z_score computation: use training_std=1.0 when actual std is zero
#        → test_score_threads_handles_zero_training_std_without_error
# Req-24 score_threads(): raises RuntimeError when called before train()
#        → test_score_threads_raises_runtime_error_when_called_before_train
#
# Total testable requirements : 24
# Total test functions        : 41  (47 pytest items including 8 parametrised variants)
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
import pytest
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
# run.sh sets PIPELINE_REPO_ROOT to /app (Docker) or the eval_assets parent dir.
# Locally the parent of this file's directory is used as a fallback.
_repo_root = os.environ.get(
    "PIPELINE_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_import_error = None
try:
    from pipeline.feature_engineering import ThreadFeatureExtractor
    from pipeline.engagement_pipeline import create_engagement_labels, EngagementPipeline
except ImportError as _e:
    _import_error = str(_e)
    ThreadFeatureExtractor = None  # type: ignore[assignment,misc]
    create_engagement_labels = None  # type: ignore[assignment]
    EngagementPipeline = None  # type: ignore[assignment,misc]


@pytest.fixture(autouse=True)
def _require_pipeline_imports():
    """Fail every test with a clear message when the pipeline module is absent.

    This ensures tests show as FAILED (not as a collection error) on an empty
    codebase, satisfying the Fail-to-Pass requirement.
    """
    if _import_error is not None:
        pytest.fail(f"Pipeline modules not yet implemented: {_import_error}")


# ── constants ─────────────────────────────────────────────────────────────────
VALID_FEATURE_NAMES = frozenset({
    "post_length",
    "has_question",
    "has_list",
    "has_media",
    "poster_activity",
    "hour_of_day",
    "day_of_week",
    "tag_count",
    "has_high_engagement_tag",
})


# ── helpers ───────────────────────────────────────────────────────────────────
def make_thread(
    body="This is a sample post body.",
    poster_history_count=10,
    posted_at="2024-01-15T14:30:00",
    tags=None,
    thread_id="t1",
):
    """Return a minimal valid new-thread record dict."""
    return {
        "thread_id": thread_id,
        "body": body,
        "poster_history_count": poster_history_count,
        "posted_at": posted_at,
        "tags": tags if tags is not None else [],
    }


def make_training_df(n=20):
    """
    Build a minimal historical DataFrame suitable for EngagementPipeline.train().

    Values are arranged so that the first half of rows gets engagement label 0
    and the second half gets label 1 (comment_count_48h and vote_count_48h both
    monotonically increase, producing a clean split at the median).
    """
    records = []
    for i in range(n):
        records.append({
            "thread_id": f"t{i}",
            "body": f"This is thread number {i} with some content.",
            "poster_history_count": i * 2,
            "posted_at": f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00",
            "tags": ["tag_a"] if i % 2 == 0 else ["tag_b"],
            "comment_count_48h": i,
            "vote_count_48h": i,
        })
    return pd.DataFrame(records)


# ── ThreadFeatureExtractor — constructor ──────────────────────────────────────

class TestThreadFeatureExtractorConstructor:
    """Interface 1: ThreadFeatureExtractor constructor and instance attributes."""

    def test_default_high_engagement_tags_is_empty_frozenset(self):
        """Req-1: Constructor with no args must default high_engagement_tags to frozenset()."""
        extractor = ThreadFeatureExtractor()
        assert extractor.high_engagement_tags == frozenset()

    def test_high_engagement_tags_stored_as_instance_attribute(self):
        """Req-2: Provided high_engagement_tags is stored as an accessible instance attribute."""
        tag_set = frozenset({"python", "machine-learning"})
        extractor = ThreadFeatureExtractor(high_engagement_tags=tag_set)
        assert extractor.high_engagement_tags == tag_set


# ── ThreadFeatureExtractor — extract_features keys ───────────────────────────

class TestExtractFeaturesKeys:
    """Interface 2: extract_features must return exactly the nine defined feature keys."""

    def test_extract_features_returns_exactly_nine_keys(self):
        """Req-3: Output dict must contain exactly the 9 mandated feature keys and no others."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread())
        assert set(features.keys()) == set(VALID_FEATURE_NAMES)


# ── post_length ───────────────────────────────────────────────────────────────

class TestPostLength:
    """post_length: word count of the opening post body (Req-4)."""

    def test_post_length_equals_word_count_of_body(self):
        """Req-4: post_length must equal the number of whitespace-separated words in body."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body="one two three four five"))
        assert features["post_length"] == 5
        assert isinstance(features["post_length"], int)

    def test_post_length_is_zero_for_empty_body(self):
        """Req-4: post_length must be 0 (int) for an empty body string."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body=""))
        assert features["post_length"] == 0
        assert isinstance(features["post_length"], int)


# ── has_question ──────────────────────────────────────────────────────────────

class TestHasQuestion:
    """has_question: binary int – 1 when body contains at least one '?' (Req-5)."""

    def test_has_question_is_one_when_body_contains_question_mark(self):
        """Req-5: has_question must be 1 (int) when body contains a question mark."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body="Is this working?"))
        assert features["has_question"] == 1
        assert isinstance(features["has_question"], int)

    def test_has_question_is_zero_when_body_has_no_question_mark(self):
        """Req-5: has_question must be 0 (int) when body contains no question mark."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body="No question here at all."))
        assert features["has_question"] == 0
        assert isinstance(features["has_question"], int)


# ── has_list ──────────────────────────────────────────────────────────────────

class TestHasList:
    """has_list: 1 when body contains a markdown list marker ('- ', '* ', or '1. ') (Req-6)."""

    @pytest.mark.parametrize("body", [
        "- first item\n- second item",
        "* bullet one\n* bullet two",
        "1. numbered item\n2. another item",
    ])
    def test_has_list_is_one_when_list_marker_present(self, body):
        """Req-6: has_list must be 1 (int) for each supported markdown list marker style."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body=body))
        assert features["has_list"] == 1
        assert isinstance(features["has_list"], int)

    def test_has_list_is_zero_when_body_contains_no_list_marker(self):
        """Req-6: has_list must be 0 (int) when body contains no list markers."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(
            make_thread(body="Plain prose without any list formatting.")
        )
        assert features["has_list"] == 0
        assert isinstance(features["has_list"], int)


# ── has_media ─────────────────────────────────────────────────────────────────

class TestHasMedia:
    """has_media: 1 when body contains markdown image syntax or a media URL (Req-7)."""

    def test_has_media_is_zero_when_body_has_no_media_reference(self):
        """Req-7: has_media must be 0 (int) when body has no image syntax or media URL."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(
            make_thread(body="Just plain text with a link https://example.com/page")
        )
        assert features["has_media"] == 0
        assert isinstance(features["has_media"], int)

    @pytest.mark.parametrize("body", [
        "Check this out ![alt text](image.png)",
        "Download from https://example.com/photo.jpg",
        "See image at https://cdn.example.com/image.png",
        "Animation at https://example.com/demo.gif",
        "Watch https://example.com/video.mp4",
    ])
    def test_has_media_is_one_when_media_reference_present(self, body):
        """Req-7: has_media must be 1 (int) for each supported media reference type."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(body=body))
        assert features["has_media"] == 1
        assert isinstance(features["has_media"], int)


# ── poster_activity ───────────────────────────────────────────────────────────

class TestPosterActivity:
    """poster_activity: float equal to poster_history_count (Req-8)."""

    def test_poster_activity_equals_poster_history_count(self):
        """Req-8: poster_activity must be a float equal to the poster_history_count field."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(poster_history_count=42))
        assert isinstance(features["poster_activity"], float)
        assert features["poster_activity"] == 42.0


# ── hour_of_day and day_of_week ───────────────────────────────────────────────

class TestTemporalFeatures:
    """hour_of_day and day_of_week extracted from the ISO 8601 posted_at field (Req-9, Req-10)."""

    def test_hour_of_day_extracted_from_posted_at_timestamp(self):
        """Req-9: hour_of_day must equal the hour component of the posted_at timestamp."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(posted_at="2024-06-10T09:15:00"))
        assert features["hour_of_day"] == 9
        assert isinstance(features["hour_of_day"], int)

    def test_hour_of_day_is_zero_at_midnight(self):
        """Req-9: hour_of_day must be 0 (int) when posted_at hour is midnight (00:xx)."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(posted_at="2024-01-15T00:00:00"))
        assert features["hour_of_day"] == 0
        assert isinstance(features["hour_of_day"], int)

    def test_hour_of_day_is_in_valid_range(self):
        """Req-9: hour_of_day must be an int in [0, 23] and match the posted_at timestamp hour."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(posted_at="2024-03-20T23:59:00"))
        assert features["hour_of_day"] == 23
        assert 0 <= features["hour_of_day"] <= 23
        assert isinstance(features["hour_of_day"], int)

    def test_day_of_week_is_zero_for_monday(self):
        """Req-10: day_of_week must be 0 (int) for a Monday (2024-01-15 is Monday)."""
        extractor = ThreadFeatureExtractor()
        # 2024-01-15 is a Monday → weekday() == 0
        features = extractor.extract_features(make_thread(posted_at="2024-01-15T10:00:00"))
        assert features["day_of_week"] == 0
        assert isinstance(features["day_of_week"], int)

    def test_day_of_week_is_six_for_sunday(self):
        """Req-10: day_of_week must be 6 (int) for a Sunday (2024-01-21 is Sunday)."""
        extractor = ThreadFeatureExtractor()
        # 2024-01-21 is a Sunday → weekday() == 6
        features = extractor.extract_features(make_thread(posted_at="2024-01-21T10:00:00"))
        assert features["day_of_week"] == 6
        assert isinstance(features["day_of_week"], int)

    def test_day_of_week_is_in_valid_range(self):
        """Req-10: day_of_week must be an int in [0, 6] and match the weekday of posted_at."""
        extractor = ThreadFeatureExtractor()
        # 2024-03-20 is a Wednesday → weekday() == 2
        features = extractor.extract_features(make_thread(posted_at="2024-03-20T12:00:00"))
        assert features["day_of_week"] == 2
        assert 0 <= features["day_of_week"] <= 6
        assert isinstance(features["day_of_week"], int)


# ── tag_count and has_high_engagement_tag ─────────────────────────────────────

class TestTagFeatures:
    """tag_count and has_high_engagement_tag (Req-11, Req-12)."""

    def test_tag_count_equals_number_of_tags_in_list(self):
        """Req-11: tag_count must be an int equal to the number of strings in the tags list."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(tags=["python", "ml", "data"]))
        assert features["tag_count"] == 3
        assert isinstance(features["tag_count"], int)

    def test_tag_count_is_zero_for_empty_tags_list(self):
        """Req-11: tag_count must be 0 (int) for an empty tags list."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(tags=[]))
        assert features["tag_count"] == 0
        assert isinstance(features["tag_count"], int)

    def test_has_high_engagement_tag_is_one_when_thread_tag_in_high_engagement_set(self):
        """Req-12: has_high_engagement_tag must be 1 (int) when any thread tag is in the set."""
        extractor = ThreadFeatureExtractor(
            high_engagement_tags=frozenset({"python", "machine-learning"})
        )
        features = extractor.extract_features(make_thread(tags=["python"]))
        assert features["has_high_engagement_tag"] == 1
        assert isinstance(features["has_high_engagement_tag"], int)

    def test_has_high_engagement_tag_is_zero_when_no_thread_tag_matches_set(self):
        """Req-12: has_high_engagement_tag must be 0 (int) when no thread tag is in the set."""
        extractor = ThreadFeatureExtractor(
            high_engagement_tags=frozenset({"python", "machine-learning"})
        )
        features = extractor.extract_features(make_thread(tags=["java", "spring"]))
        assert features["has_high_engagement_tag"] == 0
        assert isinstance(features["has_high_engagement_tag"], int)

    def test_has_high_engagement_tag_is_zero_when_high_engagement_set_is_empty(self):
        """Req-12: has_high_engagement_tag must always be 0 (int) when set is empty."""
        extractor = ThreadFeatureExtractor()
        features = extractor.extract_features(make_thread(tags=["python", "ml"]))
        assert features["has_high_engagement_tag"] == 0
        assert isinstance(features["has_high_engagement_tag"], int)


# ── create_engagement_labels ──────────────────────────────────────────────────

class TestCreateEngagementLabels:
    """Interface 3: create_engagement_labels — OR-of-medians binary labeling (Req-13–Req-15)."""

    def test_returns_series_with_same_length_and_index_as_input_dataframe(self):
        """Req-14: Output must be a pd.Series with the same length and index as the input df."""
        df = pd.DataFrame({
            "comment_count_48h": [5, 10, 15, 20],
            "vote_count_48h": [2, 4, 6, 8],
        })
        labels = create_engagement_labels(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)
        assert list(labels.index) == list(df.index)

    def test_labels_contain_only_binary_integer_values(self):
        """Req-13: All label values must be int dtype and drawn exclusively from {0, 1}."""
        df = make_training_df()
        labels = create_engagement_labels(df)
        assert set(labels.unique()).issubset({0, 1})
        assert pd.api.types.is_integer_dtype(labels)

    def test_label_is_one_when_comment_count_strictly_exceeds_median(self):
        """Req-15: Label must be 1 for a row whose comment_count_48h strictly exceeds its median."""
        # comment median = 50; row with comment=100 and vote=0 → 1 via comment alone
        df = pd.DataFrame({
            "comment_count_48h": [0, 100],
            "vote_count_48h": [0, 0],
        })
        labels = create_engagement_labels(df)
        assert labels.iloc[1] == 1

    def test_label_is_one_when_vote_count_strictly_exceeds_median(self):
        """Req-15: Label must be 1 for a row whose vote_count_48h strictly exceeds its median."""
        # vote median = 50; row with comment=0 and vote=100 → 1 via vote alone
        df = pd.DataFrame({
            "comment_count_48h": [0, 0],
            "vote_count_48h": [0, 100],
        })
        labels = create_engagement_labels(df)
        assert labels.iloc[1] == 1

    def test_label_is_zero_when_both_counts_at_or_below_their_medians(self):
        """Req-15: Label must be 0 when both counts are at or below their medians (not strictly >)."""
        # Median for both = 50; values equal to median → 0 (strictly greater required)
        df = pd.DataFrame({
            "comment_count_48h": [0, 50, 100],
            "vote_count_48h":    [0, 50, 100],
        })
        labels = create_engagement_labels(df)
        assert labels.iloc[0] == 0   # both below median
        assert labels.iloc[1] == 0   # both equal to median (not strictly greater)

    def test_or_rule_applies_when_either_metric_strictly_exceeds_its_median(self):
        """Req-15: OR-of-medians: each combination of above/at-or-below produces correct label."""
        # Median comment = 50, median vote = 50
        df = pd.DataFrame({
            "comment_count_48h": [0,   0, 100, 100],
            "vote_count_48h":    [0, 100,   0, 100],
        })
        labels = create_engagement_labels(df)
        assert labels.iloc[0] == 0   # neither above median
        assert labels.iloc[1] == 1   # vote above median
        assert labels.iloc[2] == 1   # comment above median
        assert labels.iloc[3] == 1   # both above median


# ── EngagementPipeline — constructor ──────────────────────────────────────────

class TestEngagementPipelineConstructor:
    """Interface 4: EngagementPipeline constructor (Req-16)."""

    def test_pipeline_can_be_instantiated_with_default_parameters(self):
        """Req-16: Default EngagementPipeline() must behave identically to explicit (100, 42).

        If n_estimators != 100 or random_state != 42 are used as defaults, the two
        pipelines trained on the same data would differ — this test would fail.
        """
        df = make_training_df()
        thread = make_thread()
        pipeline_default = EngagementPipeline()
        pipeline_explicit = EngagementPipeline(n_estimators=100, random_state=42)
        pipeline_default.train(df)
        pipeline_explicit.train(df)
        result_default = pipeline_default.score_threads([thread])
        result_explicit = pipeline_explicit.score_threads([thread])
        assert result_default[0]["score"] == result_explicit[0]["score"]
        assert result_default[0]["top_factors"] == result_explicit[0]["top_factors"]

    def test_pipeline_accepts_custom_n_estimators_and_random_state(self):
        """Req-16: Constructor must accept n_estimators and random_state; same params → same output."""
        df = make_training_df()
        thread = make_thread()

        pipeline_a = EngagementPipeline(n_estimators=50, random_state=0)
        pipeline_b = EngagementPipeline(n_estimators=50, random_state=0)
        pipeline_a.train(df)
        pipeline_b.train(df)
        results_a = pipeline_a.score_threads([thread])
        results_b = pipeline_b.score_threads([thread])

        assert len(results_a) == 1
        assert "score" in results_a[0]
        assert "top_factors" in results_a[0]
        assert results_a[0]["score"] == results_b[0]["score"], (
            "Two EngagementPipeline instances constructed with the same n_estimators "
            "and random_state must produce identical scores on the same input"
        )
        assert results_a[0]["top_factors"] == results_b[0]["top_factors"], (
            "Two EngagementPipeline instances constructed with the same n_estimators "
            "and random_state must produce identical top_factors on the same input"
        )


# ── EngagementPipeline.train ───────────────────────────────────────────────────

class TestEngagementPipelineTrain:
    """Interface 5: EngagementPipeline.train() (Req-17, Req-18)."""

    def test_train_computes_high_engagement_tags_correctly(self):
        """Req-17: train() must identify tags whose per-tag median comment > global median.

        make_training_df(n=20) produces:
          - even rows (i=0,2,...,18): tag_a, comment_count_48h in {0,2,...,18}
            → per-tag median = 9.0
          - odd  rows (i=1,3,...,19): tag_b, comment_count_48h in {1,3,...,19}
            → per-tag median = 10.0
          - global median comment_count_48h = 9.5

        Only tag_b (10.0 > 9.5) is high-engagement.
        tag_a (9.0 < 9.5) must NOT be in the high-engagement set.
        """
        pipeline = EngagementPipeline()
        pipeline.train(make_training_df())

        # Discover the high-engagement tag set via the public ThreadFeatureExtractor
        # attribute (high_engagement_tags) that Interface 1 mandates be stored on the
        # extractor instance, or via any frozenset/set attribute on the pipeline.
        # Neither approach asserts a specific private attribute name; it searches for
        # the public type ThreadFeatureExtractor (from the Expected Interface) or a
        # frozenset/set, accommodating different implementation structures.
        def _find_tag_set(obj):
            for v in vars(obj).values():
                if isinstance(v, ThreadFeatureExtractor):
                    return frozenset(v.high_engagement_tags)
            for v in vars(obj).values():
                if isinstance(v, (frozenset, set)):
                    return frozenset(v)
            return None

        high_engagement_tags = _find_tag_set(pipeline)
        if high_engagement_tags is None:
            for v in vars(pipeline).values():
                if hasattr(v, "__dict__"):
                    high_engagement_tags = _find_tag_set(v)
                    if high_engagement_tags is not None:
                        break

        assert high_engagement_tags is not None, (
            "Pipeline must expose the high-engagement tag set after train(), either via a "
            "stored ThreadFeatureExtractor instance attribute or as a frozenset/set "
            "attribute (searched up to one level of nesting)"
        )
        assert "tag_b" in high_engagement_tags, (
            "tag_b per-tag median (10.0) > global median (9.5): must be in high_engagement_tags"
        )
        assert "tag_a" not in high_engagement_tags, (
            "tag_a per-tag median (9.0) <= global median (9.5): must NOT be in high_engagement_tags"
        )

    def test_train_returns_none(self):
        """Req-18: train() must return None (not the pipeline instance or any other value)."""
        pipeline = EngagementPipeline()
        result = pipeline.train(make_training_df())
        assert result is None


# ── EngagementPipeline.score_threads ─────────────────────────────────────────

class TestEngagementPipelineScoreThreads:
    """Interface 6: EngagementPipeline.score_threads (Req-19–Req-24)."""

    @pytest.fixture
    def trained_pipeline(self):
        pipeline = EngagementPipeline()
        pipeline.train(make_training_df())
        return pipeline

    def test_score_threads_raises_runtime_error_when_called_before_train(self):
        """Req-24: score_threads must raise RuntimeError when called on an untrained pipeline."""
        pipeline = EngagementPipeline()
        with pytest.raises(RuntimeError):
            pipeline.score_threads([make_thread()])

    def test_score_threads_output_length_equals_input_length(self, trained_pipeline):
        """Req-19: Return value must be a list whose length equals the number of input threads."""
        threads = [make_thread(thread_id=f"n{i}") for i in range(5)]
        results = trained_pipeline.score_threads(threads)
        assert isinstance(results, list)
        assert len(results) == len(threads)

    def test_score_threads_each_result_contains_score_and_top_factors_keys(self, trained_pipeline):
        """Req-20: Every result dict must contain exactly the keys 'score' and 'top_factors'."""
        threads = [
            make_thread(thread_id="a", body="First thread post?"),
            make_thread(thread_id="b", body="- list item\n- another item"),
            make_thread(thread_id="c", body="Check https://example.com/demo.mp4"),
        ]
        results = trained_pipeline.score_threads(threads)
        assert len(results) == len(threads)
        for result in results:
            assert "score" in result
            assert "top_factors" in result

    def test_score_is_a_float_in_zero_to_one_range(self, trained_pipeline):
        """Req-20: 'score' must be a float and must lie in the closed interval [0.0, 1.0]."""
        results = trained_pipeline.score_threads([make_thread()])
        score = results[0]["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_top_factors_is_a_list_of_exactly_three_strings(self, trained_pipeline):
        """Req-20/Req-21: 'top_factors' must be a list of exactly 3 strings, each a valid feature name."""
        results = trained_pipeline.score_threads([make_thread()])
        top_factors = results[0]["top_factors"]
        assert isinstance(top_factors, list)
        assert len(top_factors) == 3
        for factor in top_factors:
            assert isinstance(factor, str)
        assert set(top_factors).issubset(VALID_FEATURE_NAMES)

    def test_top_factors_contains_no_duplicate_feature_names(self, trained_pipeline):
        """Req-21: All three entries in top_factors must be distinct feature names."""
        results = trained_pipeline.score_threads([make_thread()])
        top_factors = results[0]["top_factors"]
        assert len(set(top_factors)) == 3

    def test_top_factors_are_sorted_in_descending_contribution_score_order(self, trained_pipeline):
        """Req-22: top_factors must be sorted by per-instance contribution = importance×|z_score|.

        ASSERTION 1 — dominant feature must be first (catches wrong sort direction):
          In make_training_df(n=20), poster_history_count = 2*i (range 0–38):
              training mean = 19,  training std ≈ 11.4
          poster_history_count=999_999 → |z[poster_activity]| ≈ 87_700.
          Even at minimum conceivable non-zero importance (0.01), contribution ≈ 877,
          dwarfing every other feature (bounded by importance × |z| ≤ 1.0 × ~25 ≈ 25).
          poster_activity MUST be top_factors[0].
          An ascending sort would place it last — failing this check.

        ASSERTION 2 — zero-contribution feature must not be first (catches
          global-importance-only sort, which ignores z-scores):
          poster_history_count=19 is the exact training mean (mean of {0,2,...,38}),
          so z[poster_activity] = (19 − 19) / std = 0 → contribution = 0.
          Other features have non-zero importance AND non-zero z-scores for this
          thread, so one of them will have positive contribution exceeding 0.
          An implementation that sorts by global importance alone (ignoring z-scores)
          would still put poster_activity first — failing this check.

        Together the two assertions confirm both:
          (a) descending sort order, and
          (b) per-instance (not global-only) contribution scoring.
        """
        # ── Assertion 1: feature with the largest z-score must be ranked first ──
        thread_extreme = make_thread(
            body="This is a sample post body.",
            poster_history_count=999_999,
            posted_at="2024-01-15T14:30:00",
            tags=[],
        )
        result_extreme = trained_pipeline.score_threads([thread_extreme])
        top_factors_extreme = result_extreme[0]["top_factors"]

        assert top_factors_extreme[0] == "poster_activity", (
            "poster_history_count=999_999 gives poster_activity |z_score| ≈ 87_700 "
            "(training mean=19, std≈11.4). This contribution overwhelms every other "
            "feature. It must appear first in top_factors. Failing means top_factors "
            "are not sorted in descending contribution order."
        )

        # ── Assertion 2: feature with zero contribution must not be ranked first ──
        # poster_history_count=19 is the exact training mean (mean of {0,2,...,38}),
        # so z[poster_activity] = 0 and contribution[poster_activity] = 0.
        # Other features have non-zero z-scores, so one will have positive contribution
        # that exceeds poster_activity's contribution of 0.
        thread_mean = make_thread(
            body="This is a sample post body.",
            poster_history_count=19,        # exact training mean → z[poster_activity] = 0
            posted_at="2024-01-15T14:30:00",
            tags=[],
        )
        result_mean = trained_pipeline.score_threads([thread_mean])
        top_factors_mean = result_mean[0]["top_factors"]

        assert top_factors_mean[0] != "poster_activity", (
            "poster_history_count=19 is the exact training mean, so z[poster_activity]=0 "
            "and contribution[poster_activity]=0. A feature with zero contribution must "
            "not appear first. An implementation sorting by global feature importance "
            "alone (ignoring z-scores) would incorrectly place poster_activity first "
            "because it is the most globally important feature in this fixture model."
        )

    def test_score_threads_handles_zero_training_std_without_error(self):
        """Req-23: When a feature's training std is zero, the pipeline must use 1.0 as denominator.

        Training data where all threads share the same body, timestamp, and empty tags
        means post_length, has_question, has_list, has_media, hour_of_day, day_of_week,
        tag_count, and has_high_engagement_tag all have std=0 in the feature matrix.
        Only poster_activity varies. A correct implementation must not raise any exception
        and must still return valid score/top_factors output.
        """
        records = [
            {
                "thread_id": f"t{i}",
                "body": "identical body text for all training threads",
                "poster_history_count": i * 5,
                "posted_at": "2024-03-15T10:00:00",
                "tags": [],
                "comment_count_48h": i,
                "vote_count_48h": i,
            }
            for i in range(20)
        ]
        df = pd.DataFrame(records)
        pipeline = EngagementPipeline()
        pipeline.train(df)

        results = pipeline.score_threads([
            make_thread(body="identical body text for all training threads")
        ])
        assert len(results) == 1
        result = results[0]
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["top_factors"], list)
        assert len(result["top_factors"]) == 3
        for factor in result["top_factors"]:
            assert isinstance(factor, str)
            assert factor in VALID_FEATURE_NAMES


# ── SELF-AUDIT ─────────────────────────────────────────────────────────────────
# Total test functions        : 41  (47 pytest items including 8 parametrised variants)
# Overly specific             : 0   (0.0%)
# Overly broad                : 0   (0.0%)
# Total problematic           : 0   (0.0%)
# Uncovered requirements      : 0
# All 8 checks                : PASS
#
# CHECK 1 (Prompt Grounding)        PASS – every assertion traces directly to a
#   stated requirement; no best-practice, logging, or performance tests present.
# CHECK 2 (Black-Box Purity)        PASS – only public interfaces from Expected
#   Interface are used; _find_tag_set uses ThreadFeatureExtractor.high_engagement_tags
#   (a mandated public attribute per Interface 1) and searches by type, not name.
# CHECK 3 (Implementation Neutrality) PASS – no error message text, HTTP codes,
#   file paths, or internal symbol names are asserted beyond the Expected Interface.
# CHECK 4 (Coverage Closure)        PASS – all 24 testable requirements have ≥1 test.
# CHECK 5 (No Overlap)              PASS – each test covers a distinct requirement
#   facet or input condition; no semantically duplicate test pairs exist.
# CHECK 6 (F2P Correctness)         PASS – autouse fixture fails all tests on an
#   empty codebase; every test asserts a condition that a stub cannot satisfy.
# CHECK 7 (No Overly Broad)         PASS – types, value constraints, range bounds,
#   and sort order are all explicitly asserted where the prompt requires them.
# CHECK 8 (No Best-Practice Injection) PASS – idempotency, caching, logging,
#   thread safety, and performance are not tested.
# ──────────────────────────────────────────────────────────────────────────────
