from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


try:
    from gensim.models import Word2Vec
except Exception:  # pragma: no cover - optional dependency
    Word2Vec = None


LABEL_TO_ID = {"FAKE": 0, "REAL": 1}
ID_TO_LABEL = {0: "FAKE", 1: "REAL"}


@dataclass
class TrainingArtifacts:
    vectorizer: TfidfVectorizer
    models: Dict[str, object]
    metrics: pd.DataFrame
    best_model_name: str
    dataset_source: str
    advanced_model_name: Optional[str]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def evaluate_model(model: object, X_test, y_test) -> Dict[str, float]:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def train_classic_models(
    texts: List[str],
    labels: List[int],
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, Dict[str, object], pd.DataFrame]:
    vectorizer = build_tfidf_vectorizer()
    features = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
        ),
    }

    rows = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["f1_score", "accuracy", "precision", "recall"],
        ascending=False,
    )
    return vectorizer, trained_models, metrics_df.reset_index(drop=True)


def train_word2vec_model(
    texts: List[str],
    labels: List[int],
    random_state: int = 42,
) -> Tuple[Optional[object], Optional[Dict[str, object]]]:
    if Word2Vec is None:
        return None, None

    tokenized = [text.split() for text in texts if text.strip()]
    if not tokenized:
        return None, None

    word2vec = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        seed=random_state,
        epochs=40,
    )

    def embed(document: str) -> np.ndarray:
        vectors = [word2vec.wv[token] for token in document.split() if token in word2vec.wv]
        if not vectors:
            return np.zeros(word2vec.vector_size, dtype=float)
        return np.mean(vectors, axis=0)

    features = np.vstack([embed(text) for text in texts])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    model = LogisticRegression(max_iter=1500, random_state=random_state)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, {
        "word2vec_model": word2vec,
        "embedding_classifier": model,
        "metrics": metrics,
    }


def select_best_model(metrics_df: pd.DataFrame, advanced_result: Optional[Dict[str, object]]) -> str:
    best_row = metrics_df.iloc[0]
    best_name = str(best_row["model"])
    best_f1 = float(best_row["f1_score"])
    best_accuracy = float(best_row["accuracy"])

    if advanced_result:
        advanced_metrics = advanced_result["metrics"]
        advanced_f1 = float(advanced_metrics["f1_score"])
        advanced_accuracy = float(advanced_metrics["accuracy"])
        if (advanced_f1, advanced_accuracy) > (best_f1, best_accuracy):
            return "Word2Vec Logistic Regression"

    return best_name


def train_all_models(
    df: pd.DataFrame,
    dataset_source: str,
    enable_word2vec: bool = False,
    random_state: int = 42,
) -> TrainingArtifacts:
    labels = [LABEL_TO_ID[label] for label in df["label"].tolist()]
    texts = df["clean_text"].tolist()

    vectorizer, models, metrics_df = train_classic_models(texts, labels, random_state=random_state)

    advanced_result = None
    advanced_model_name = None
    if enable_word2vec:
        advanced_model, advanced_result = train_word2vec_model(texts, labels, random_state=random_state)
        if advanced_model is not None and advanced_result is not None:
            models["Word2Vec Logistic Regression"] = advanced_model
            advanced_model_name = "Word2Vec Logistic Regression"
            metrics_df = pd.concat(
                [
                    metrics_df,
                    pd.DataFrame(
                        [{"model": advanced_model_name, **advanced_result["metrics"]}]
                    ),
                ],
                ignore_index=True,
            ).sort_values(
                by=["f1_score", "accuracy", "precision", "recall"],
                ascending=False,
            ).reset_index(drop=True)

    best_model_name = select_best_model(metrics_df, advanced_result)

    return TrainingArtifacts(
        vectorizer=vectorizer,
        models=models,
        metrics=metrics_df,
        best_model_name=best_model_name,
        dataset_source=dataset_source,
        advanced_model_name=advanced_model_name,
    )


def save_training_artifacts(artifacts: TrainingArtifacts, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bundle_path = output_path / "model_bundle.pkl"
    summary_path = output_path / "training_summary.json"
    metrics_path = output_path / "metrics_summary.csv"

    serializable_bundle = {
        "vectorizer": artifacts.vectorizer,
        "models": artifacts.models,
        "best_model_name": artifacts.best_model_name,
        "dataset_source": artifacts.dataset_source,
        "advanced_model_name": artifacts.advanced_model_name,
        "metrics": artifacts.metrics.to_dict(orient="records"),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
    }

    joblib.dump(serializable_bundle, bundle_path)
    artifacts.metrics.to_csv(metrics_path, index=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model_name": artifacts.best_model_name,
                "dataset_source": artifacts.dataset_source,
                "models_trained": list(artifacts.models.keys()),
                "advanced_model_name": artifacts.advanced_model_name,
            },
            file,
            indent=2,
        )

    return bundle_path


def load_model_bundle(bundle_path: str | Path) -> Dict[str, object]:
    return joblib.load(bundle_path)
