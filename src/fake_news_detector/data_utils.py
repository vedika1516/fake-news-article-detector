from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .text_processing import merge_text_fields, preprocess_text


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

FAKE_LABELS = {"fake", "false", "pants-fire", "pants on fire", "barely-true"}
REAL_LABELS = {"real", "true", "mostly-true", "half-true"}


def normalize_label(value: str) -> str | None:
    if pd.isna(value):
        return None

    label = str(value).strip().lower()
    if label in FAKE_LABELS or label == "0":
        return "FAKE"
    if label in REAL_LABELS or label == "1":
        return "REAL"
    return None


def _load_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    available = set(df.columns.str.lower())

    if {"title", "text", "label"}.issubset(available):
        renamed = {column: column.lower() for column in df.columns}
        df = df.rename(columns=renamed)
        return df[["title", "text", "label"]].copy()

    if {"text", "label"}.issubset(available):
        renamed = {column: column.lower() for column in df.columns}
        df = df.rename(columns=renamed)
        df["title"] = ""
        return df[["title", "text", "label"]].copy()

    raise ValueError(
        "Unsupported CSV format. Expected columns like title, text, and label."
    )


def _load_kaggle_folder(path: Path) -> pd.DataFrame:
    fake_path = path / "Fake.csv"
    true_path = path / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise ValueError("Kaggle dataset folder must contain Fake.csv and True.csv.")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    for frame, label in ((fake_df, "FAKE"), (true_df, "REAL")):
        if "text" not in frame.columns:
            raise ValueError("Kaggle files must contain a text column.")
        if "title" not in frame.columns:
            frame["title"] = ""
        frame["label"] = label

    combined = pd.concat(
        [fake_df[["title", "text", "label"]], true_df[["title", "text", "label"]]],
        ignore_index=True,
    )
    return combined


def _load_liar_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] == len(LIAR_COLUMNS):
        df.columns = LIAR_COLUMNS
    else:
        raise ValueError("Unexpected LIAR TSV format.")

    df["title"] = ""
    df["text"] = df["statement"]
    return df[["title", "text", "label"]].copy()


def _load_liar_folder(path: Path) -> pd.DataFrame:
    frames = []
    for file_name in ("train.tsv", "test.tsv", "valid.tsv"):
        candidate = path / file_name
        if candidate.exists():
            frames.append(_load_liar_file(candidate))
    if not frames:
        raise ValueError("No LIAR TSV files found in the provided folder.")
    return pd.concat(frames, ignore_index=True)


def load_dataset(path: str | Path) -> Tuple[pd.DataFrame, str]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.is_dir():
        if (dataset_path / "Fake.csv").exists():
            raw_df = _load_kaggle_folder(dataset_path)
            source_name = "Kaggle Fake and True News"
        else:
            raw_df = _load_liar_folder(dataset_path)
            source_name = "LIAR"
    elif dataset_path.suffix.lower() == ".tsv":
        raw_df = _load_liar_file(dataset_path)
        source_name = "LIAR"
    else:
        raw_df = _load_single_csv(dataset_path)
        source_name = "Custom CSV"

    raw_df["title"] = raw_df["title"].fillna("")
    raw_df["text"] = raw_df["text"].fillna("")
    raw_df["label"] = raw_df["label"].apply(normalize_label)
    raw_df = raw_df.dropna(subset=["label"]).copy()

    raw_df["combined_text"] = raw_df.apply(
        lambda row: merge_text_fields(row["title"], row["text"]),
        axis=1,
    )
    raw_df["clean_text"] = raw_df["combined_text"].apply(preprocess_text)
    raw_df = raw_df[raw_df["clean_text"].str.len() > 0].copy()

    if raw_df.empty:
        raise ValueError("Dataset is empty after preprocessing and label normalization.")

    return raw_df.reset_index(drop=True), source_name
