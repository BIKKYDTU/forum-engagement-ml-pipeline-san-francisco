# Forum Thread Engagement Prediction Pipeline

A Python machine learning pipeline that predicts which new forum threads will become engaging, enabling moderators to prioritize highlighting them. The pipeline engineers structured features from post content, poster history, and timing, trains a Random Forest classifier on historical engagement data, and produces a per-thread engagement probability score along with the top contributing feature factors for every new thread.

---

## Table of Contents

- [Overview](#overview)
- [Features Engineered](#features-engineered)
- [Engagement Labeling](#engagement-labeling)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Docker Usage](#docker-usage)
- [Tech Stack](#tech-stack)

---

## Overview

This pipeline solves the problem of identifying high-potential forum threads early. Given a dataset of historical threads labeled by their 48-hour engagement (comment and vote counts), the pipeline:

1. **Engineers nine features** from each thread record (content signals, poster history, temporal patterns, and tag metadata).
2. **Labels historical threads** using an OR-of-medians rule on comment and vote counts.
3. **Trains a Random Forest classifier** on the feature matrix.
4. **Scores new threads** — returning a probability score in `[0.0, 1.0]` and a ranked list of the top 3 contributing features for each thread, computed per-instance rather than globally.

---

## Features Engineered

All nine features are extracted from a thread record by `ThreadFeatureExtractor`:

| Feature | Type | Description |
|---|---|---|
| `post_length` | int | Word count of the opening post body |
| `has_question` | int | `1` if the body contains at least one `?`, else `0` |
| `has_list` | int | `1` if the body contains a markdown list marker (`- `, `* `, or `1. `), else `0` |
| `has_media` | int | `1` if the body contains `![` (markdown image) or a URL ending in `.jpg`, `.png`, `.gif`, `.mp4`, else `0` |
| `poster_activity` | float | Poster's number of prior posts (`poster_history_count`) |
| `hour_of_day` | int | Hour (0–23) extracted from the ISO 8601 `posted_at` timestamp |
| `day_of_week` | int | Day of week (0 = Monday, 6 = Sunday) extracted from `posted_at` |
| `tag_count` | int | Number of topic tags attached to the thread |
| `has_high_engagement_tag` | int | `1` if any thread tag belongs to the high-engagement tag set derived during training, else `0` |

> **High-engagement tags** are computed during `train()`: a tag is considered high-engagement if the median `comment_count_48h` of training threads carrying that tag is **strictly greater** than the global median `comment_count_48h` across all training threads.

---

## Engagement Labeling

A binary label is derived for each historical thread using the **OR-of-medians** rule:

```
label = 1  if  comment_count_48h > median(comment_count_48h)
              OR  vote_count_48h > median(vote_count_48h)
label = 0  otherwise
```

This is implemented by `create_engagement_labels(df)` in `pipeline/engagement_pipeline.py`.

---

## Project Structure

```
.
├── pipeline/
│   ├── __init__.py
│   ├── feature_engineering.py     # ThreadFeatureExtractor — extracts 9 features per thread
│   └── engagement_pipeline.py     # create_engagement_labels + EngagementPipeline
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py           # Full test suite covering all interfaces
├── Dockerfile                     # Ubuntu 22.04 container with all dependencies
├── requirements.txt               # Python dependencies
├── run.sh                         # Test runner script
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/BIKKYDTU/forum-engagement-ml-pipeline-san-francisco.git
cd forum-engagement-ml-pipeline-san-francisco

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
pytest
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.3,<2.0
```

---

## Quick Start

```python
import pandas as pd
from pipeline.engagement_pipeline import EngagementPipeline

# ── 1. Prepare historical training data ──────────────────────────────────────
historical_data = pd.DataFrame([
    {
        "thread_id": "t1",
        "body": "How do I get started with Python? Here are my issues:\n- Can't install pip\n- Import errors",
        "poster_history_count": 5,
        "posted_at": "2024-01-15T09:00:00",
        "tags": ["python", "beginners"],
        "comment_count_48h": 42,
        "vote_count_48h": 18,
    },
    {
        "thread_id": "t2",
        "body": "Just sharing a thought.",
        "poster_history_count": 1,
        "posted_at": "2024-01-16T02:00:00",
        "tags": ["off-topic"],
        "comment_count_48h": 1,
        "vote_count_48h": 0,
    },
    # ... more training rows
])

# ── 2. Train the pipeline ─────────────────────────────────────────────────────
pipeline = EngagementPipeline(n_estimators=100, random_state=42)
pipeline.train(historical_data)

# ── 3. Score new threads ──────────────────────────────────────────────────────
new_threads = [
    {
        "thread_id": "new_1",
        "body": "Is anyone else struggling with async/await in JavaScript? ![screenshot](https://example.com/err.png)",
        "poster_history_count": 120,
        "posted_at": "2024-02-01T14:30:00",
        "tags": ["javascript", "async"],
    },
]

results = pipeline.score_threads(new_threads)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Top factors: {result['top_factors']}")
# Score: 0.7800
# Top factors: ['poster_activity', 'has_media', 'has_question']
```

---

## API Reference

### `ThreadFeatureExtractor`
**Module:** `pipeline/feature_engineering.py`

Transforms a single thread record dict into nine engineered features.

```python
from pipeline.feature_engineering import ThreadFeatureExtractor

extractor = ThreadFeatureExtractor(high_engagement_tags=frozenset({"python", "ml"}))
features = extractor.extract_features(thread_dict)
# Returns: {'post_length': 12, 'has_question': 1, 'has_list': 0, ...}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `high_engagement_tags` | `frozenset[str]` | `frozenset()` | Tag strings classified as high-engagement from training. When empty, `has_high_engagement_tag` is always `0`. |

---

### `create_engagement_labels(df)`
**Module:** `pipeline/engagement_pipeline.py`

Computes binary engagement labels for a training DataFrame using the OR-of-medians rule.

```python
from pipeline.engagement_pipeline import create_engagement_labels

labels = create_engagement_labels(df)  # Returns pd.Series of int (0 or 1)
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Training DataFrame with `comment_count_48h` and `vote_count_48h` columns |

**Returns:** `pd.Series` of dtype `int`, values in `{0, 1}`, same index as `df`.

---

### `EngagementPipeline`
**Module:** `pipeline/engagement_pipeline.py`

End-to-end pipeline composing `ThreadFeatureExtractor` and a `RandomForestClassifier`.

```python
from pipeline.engagement_pipeline import EngagementPipeline

pipeline = EngagementPipeline(n_estimators=100, random_state=42)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | `int` | `100` | Number of trees in the Random Forest |
| `random_state` | `int` | `42` | Seed for reproducibility |

#### `pipeline.train(historical_data)`

Fits the classifier on historical thread data.

```python
pipeline.train(historical_data_df)  # Returns None
```

Steps performed internally:
1. Derives engagement labels via `create_engagement_labels`.
2. Computes high-engagement tags from the training set.
3. Builds the 9-feature matrix across all training rows.
4. Stores per-feature column-wise mean and standard deviation.
5. Fits the `RandomForestClassifier`.

#### `pipeline.score_threads(new_threads)`

Scores a list of new thread dicts. Raises `RuntimeError` if called before `train()`.

```python
results = pipeline.score_threads(new_threads_list)
```

**Input:** List of dicts, each with keys: `thread_id`, `body`, `poster_history_count`, `posted_at`, `tags`.

**Output:** List of dicts, one per input thread:

| Key | Type | Description |
|---|---|---|
| `score` | `float` | Predicted engagement probability in `[0.0, 1.0]` |
| `top_factors` | `list[str]` | Top 3 feature names ranked by per-instance contribution score |

**Per-instance contribution score formula:**

```
contribution[i] = feature_importances_[i] × |z_score[i]|

where z_score[i] = (feature_value[i] − training_mean[i]) / training_std[i]
      (training_std[i] is clamped to 1.0 when it equals 0)
```

This produces a distinct ranked explanation for each thread that reflects both the model's learned feature weights and how unusually that thread's features deviate from the training distribution.

---

## Running Tests

```bash
# From the repository root
pytest tests/ -v
```

The test suite covers:
- `ThreadFeatureExtractor` constructor and all nine feature extractions
- `create_engagement_labels` OR-of-medians correctness
- `EngagementPipeline` constructor parameter handling
- `train()` high-engagement tag computation
- `score_threads()` output structure, score range, `top_factors` validity
- Descending contribution sort order (per-instance, not global-importance-only)
- Zero-std feature handling (no division-by-zero errors)
- `RuntimeError` on scoring before training

---

## Docker Usage

Build and run the environment:

```bash
# Build the image
docker build -t forum-engagement-pipeline .

# Run an interactive shell
docker run -it --rm forum-engagement-pipeline

# Mount local code and run tests inside the container
docker run -it --rm \
  -v $(pwd)/pipeline:/app/pipeline \
  -v $(pwd)/tests:/app/tests \
  -e PIPELINE_REPO_ROOT=/app \
  forum-engagement-pipeline \
  bash -c "cd /app && pytest tests/ -v"
```

The `Dockerfile` uses `ubuntu:22.04` and pre-installs all Python dependencies at build time.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| pandas | `>=2.0,<3.0` | DataFrame manipulation for training data |
| scikit-learn | `>=1.3,<2.0` | `RandomForestClassifier` |
| numpy | `>=1.24,<2.0` | Feature matrix operations and z-score computation |
| pytest | latest | Test suite runner |

---

## Data Schema

| Field | Type | Present In | Description |
|---|---|---|---|
| `thread_id` | str | Training + New | Unique thread identifier |
| `body` | str | Training + New | Opening post text |
| `poster_history_count` | int | Training + New | Poster's prior post count |
| `posted_at` | str | Training + New | ISO 8601 datetime string (e.g. `"2024-01-15T14:30:00"`) |
| `tags` | list[str] | Training + New | List of topic tag strings |
| `comment_count_48h` | int | Training only | Number of comments at 48 hours |
| `vote_count_48h` | int | Training only | Number of votes at 48 hours |

---

## License

MIT
