# Forum Thread Engagement Prediction Pipeline

## Description

Build a Python machine learning pipeline that predicts which new forum threads will become engaging, so that moderators can prioritize highlighting them. The pipeline ingests historical thread data, engineers structured features from each thread record, trains a binary Random Forest classifier using engagement labels derived from 48-hour comment and vote counts, and produces a per-thread engagement probability score along with the top contributing feature factors — computed individually for each thread using per-instance feature contributions — for every new thread submitted for scoring.

## Tech Stack

- Python 3.10+
- pandas for data handling
- scikit-learn for feature processing and classification
- numpy for numerical operations

## Key Requirements

### Feature Engineering

Extract the following nine features from each thread record:

| Feature Name               | Type  | Description                                                                                   |
|----------------------------|-------|-----------------------------------------------------------------------------------------------|
| `post_length`              | int   | Word count of the opening post body                                                           |
| `has_question`             | int   | Binary indicator (1 or 0): the opening post body contains at least one question mark         |
| `has_list`                 | int   | Binary indicator (1 or 0): the opening post body contains a markdown list marker (`- ` or `* `) or a numbered list (`1. `) |
| `has_media`                | int   | Binary indicator (1 or 0): the opening post body contains an image or video reference (markdown image syntax `![` or a URL ending in `.jpg`, `.png`, `.gif`, `.mp4`) |
| `poster_activity`          | float | Poster's number of prior posts (historical activity count)                                   |
| `hour_of_day`              | int   | Hour (0–23) extracted from the posting ISO 8601 timestamp                                    |
| `day_of_week`              | int   | Day of week (0 = Monday, 6 = Sunday) extracted from the posting timestamp                    |
| `tag_count`                | int   | Count of topic tags attached to the thread                                                    |
| `has_high_engagement_tag`  | int   | Binary indicator (1 or 0): at least one of the thread's tags belongs to the set of high-engagement tags derived from training data. A tag is considered high-engagement if the median `comment_count_48h` of training threads carrying that tag is strictly greater than the global median `comment_count_48h` across all training threads. This set is computed during `train()` and supplied to the extractor; when the set is empty (e.g. before training), this feature is always `0`. |

### Engagement Labeling

Define a binary engagement label for each historical thread record using the following rule: a thread is labeled `1` (engaged) when its `comment_count_48h` is strictly greater than the median `comment_count_48h` of the full training set OR its `vote_count_48h` is strictly greater than the median `vote_count_48h` of the full training set; otherwise it is labeled `0`. Medians are computed from the training DataFrame passed to `train()`.

### Model Training

- Train a `RandomForestClassifier` from scikit-learn on the nine engineered features and the derived engagement labels.
- Accept configurable constructor parameters `n_estimators` (default: `100`) and `random_state` (default: `42`).
- After training, store the per-feature training mean and standard deviation of the feature matrix (as numpy arrays of length 9) on the pipeline instance. These statistics are used during scoring to compute per-instance contribution scores for `top_factors`.
- After training, expose global feature importances through the classifier's `feature_importances_` attribute for use during scoring.

### Scoring New Threads

- Accept a list of new thread record dicts using the new-thread schema, each containing: `thread_id`, `body`, `poster_history_count`, `posted_at`, and `tags`.
- For each new thread, return a result dict containing:
  - `score`: `float` in the range `[0.0, 1.0]` — the predicted engagement probability (positive-class probability from `predict_proba`)
  - `top_factors`: `list[str]` of exactly 3 feature names — the top 3 features ranked in descending order by that thread's **per-instance contribution scores**. The per-instance contribution score for feature `i` of a given thread is computed as: `global_importance[i] × |z_score[i]|`, where `global_importance[i]` is the classifier's `feature_importances_[i]` and `z_score[i] = (feature_value[i] − training_mean[i]) / training_std[i]` (use `training_std[i] = 1.0` when the training standard deviation is zero to avoid division by zero). The training mean and standard deviation for each feature are computed from the feature matrix built during `train()` and stored on the pipeline instance. This produces a distinct ranked explanation for each scored thread that reflects both the model's learned feature weights and how unusually that specific thread's features deviate from the training distribution.
- Raise a `RuntimeError` when `score_threads()` is called before `train()` has been called on the pipeline instance.

### Data Schema

The following table defines the complete field schema, specifying which fields are present in historical training records and which are present in new thread input records:

| Field                  | Type        | Present In          | Description                                      |
|------------------------|-------------|---------------------|--------------------------------------------------|
| `thread_id`            | str         | Training + New      | Unique thread identifier                         |
| `body`                 | str         | Training + New      | Opening post text                                |
| `poster_history_count` | int         | Training + New      | Poster's prior post count                        |
| `posted_at`            | str         | Training + New      | ISO 8601 datetime string (e.g. `"2024-01-15T14:30:00"`) |
| `tags`                 | list\[str\] | Training + New      | List of topic tag strings                        |
| `comment_count_48h`    | int         | Training only       | Number of comments at 48 hours                   |
| `vote_count_48h`       | int         | Training only       | Number of votes at 48 hours                      |

## Expected Interface

### Interface 1: ThreadFeatureExtractor

- **Path:** `pipeline/feature_engineering.py`
- **Name:** `ThreadFeatureExtractor`
- **Type:** class
- **Input:** Constructor accepts one optional parameter: `high_engagement_tags: frozenset[str] = frozenset()` — the set of tag strings identified as high-engagement from training data. When omitted or empty, `has_high_engagement_tag` is always `0`.
- **Output:** N/A
- **Bases / Overrides:** bases: object
- **Description:** Transformer class that extracts the nine engineered features from a single thread record dict. Stores `high_engagement_tags` as an instance attribute for use in `extract_features`. Used directly by the pipeline and by the test suite.

---

### Interface 2: ThreadFeatureExtractor.extract_features

- **Path:** `pipeline/feature_engineering.py`
- **Name:** `ThreadFeatureExtractor.extract_features`
- **Type:** method
- **Input:** `thread: dict` — a single thread record dict containing at minimum the fields `body` (str), `poster_history_count` (int), `posted_at` (str), and `tags` (list\[str\])
- **Output:** `dict` — flat dict with exactly the keys: `post_length`, `has_question`, `has_list`, `has_media`, `poster_activity`, `hour_of_day`, `day_of_week`, `tag_count`, `has_high_engagement_tag`
- **Bases / Overrides:** overrides: none
- **Description:** Extracts and returns all nine features for the provided thread record. Tests assert the presence of all nine feature keys and verify correct values given known inputs (e.g. a body with a question mark yields `has_question == 1`, a body without yields `has_question == 0`; a thread whose `tags` list contains a tag present in `high_engagement_tags` yields `has_high_engagement_tag == 1`, a thread with no matching tags yields `has_high_engagement_tag == 0`).

---

### Interface 3: create_engagement_labels

- **Path:** `pipeline/engagement_pipeline.py`
- **Name:** `create_engagement_labels`
- **Type:** function
- **Input:** `df: pd.DataFrame` — training DataFrame with columns `comment_count_48h` (int) and `vote_count_48h` (int)
- **Output:** `pd.Series` of dtype int with values 0 or 1, same index as `df`
- **Bases / Overrides:** N/A
- **Description:** Computes and returns the binary engagement label for each row in `df` using the OR-of-medians rule described in the Key Requirements. Tests assert that the returned Series has the same length and index as `df`, contains only integer values 0 and 1, and that a row with both `comment_count_48h` and `vote_count_48h` above their respective medians is labeled `1`, while a row with both values at or below their respective medians is labeled `0`.

---

### Interface 4: EngagementPipeline

- **Path:** `pipeline/engagement_pipeline.py`
- **Name:** `EngagementPipeline`
- **Type:** class
- **Input:** Constructor accepts `n_estimators: int = 100` and `random_state: int = 42`
- **Output:** N/A
- **Bases / Overrides:** bases: object
- **Description:** End-to-end pipeline that composes `ThreadFeatureExtractor` and a `RandomForestClassifier`. Maintains internal state after `train()` is called, including the fitted classifier, the set of high-engagement tags, and the per-feature training mean and standard deviation arrays used for per-instance `top_factors` computation. The test suite instantiates this class, calls `train()` with historical data, and then calls `score_threads()` to verify output structure and correctness.

---

### Interface 5: EngagementPipeline.train

- **Path:** `pipeline/engagement_pipeline.py`
- **Name:** `EngagementPipeline.train`
- **Type:** method
- **Input:** `historical_data: pd.DataFrame` — DataFrame with all training schema columns including `comment_count_48h` and `vote_count_48h`
- **Output:** `None`
- **Bases / Overrides:** overrides: none
- **Description:** Applies `create_engagement_labels` to derive labels, computes the set of high-engagement tags from the training DataFrame (tags whose per-tag median `comment_count_48h` exceeds the global median `comment_count_48h` of the training set), instantiates a `ThreadFeatureExtractor` configured with that tag set, runs `extract_features` on every row to build the feature matrix, stores the per-feature column-wise mean and standard deviation of that matrix, and fits the `RandomForestClassifier`. After this method returns, the pipeline is ready to call `score_threads()`.

---

### Interface 6: EngagementPipeline.score_threads

- **Path:** `pipeline/engagement_pipeline.py`
- **Name:** `EngagementPipeline.score_threads`
- **Type:** method
- **Input:** `new_threads: list[dict]` — list of thread record dicts matching the new-thread schema, each containing: `thread_id` (str), `body` (str), `poster_history_count` (int), `posted_at` (str), and `tags` (list[str])
- **Output:** `list[dict]` — one result dict per input thread, each containing keys `score` (float) and `top_factors` (list of 3 str)
- **Bases / Overrides:** overrides: none
- **Description:** Scores each new thread using the trained classifier. Returns the positive-class predicted probability as `score` and the top 3 feature names by per-instance contribution score as `top_factors`. The per-instance contribution score for feature `i` of a given thread is `global_importance[i] × |z_score[i]|` where `z_score[i] = (feature_value[i] − training_mean[i]) / training_std[i]` (clamped to `training_std[i] = 1.0` when zero). Raises a `RuntimeError` when called on an untrained pipeline instance. Tests assert: output length equals input length, every `score` is a float in `[0.0, 1.0]`, every `top_factors` is a list of exactly 3 strings each drawn from the set of the nine valid feature names (`post_length`, `has_question`, `has_list`, `has_media`, `poster_activity`, `hour_of_day`, `day_of_week`, `tag_count`, `has_high_engagement_tag`), the factors within each result are sorted in descending order of that thread's per-instance contribution scores, and calling this method on an untrained pipeline raises a `RuntimeError`.

## Current State

Empty repository. Implement the full pipeline from scratch.
