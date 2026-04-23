from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _safe_predict_proba(model, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[0]

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        positive = 1.0 / (1.0 + np.exp(-score))
        return np.array([1 - positive, positive])

    prediction = int(model.predict(features)[0])
    return np.array([1.0, 0.0]) if prediction == 0 else np.array([0.0, 1.0])


def predict_with_details(bundle: Dict[str, object], raw_text: str, model_name: str | None = None) -> Dict[str, object]:
    vectorizer = bundle["vectorizer"]
    models = bundle["models"]
    active_model_name = model_name or bundle["best_model_name"]
    model = models[active_model_name]

    features = vectorizer.transform([raw_text])
    probabilities = _safe_predict_proba(model, features)
    predicted_id = int(np.argmax(probabilities))
    predicted_label = bundle["id_to_label"][predicted_id]

    return {
        "model_name": active_model_name,
        "predicted_label": predicted_label,
        "prob_fake": round(float(probabilities[0]), 4),
        "prob_real": round(float(probabilities[1]), 4),
        "confidence": round(float(np.max(probabilities)), 4),
        "features": features,
    }


def important_terms(bundle: Dict[str, object], model_name: str, features, top_k: int = 8) -> List[Tuple[str, float]]:
    vectorizer = bundle["vectorizer"]
    model = bundle["models"][model_name]
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = features.toarray()[0]
    non_zero = np.where(row > 0)[0]

    if len(non_zero) == 0:
        return []

    if hasattr(model, "coef_"):
        contributions = row[non_zero] * model.coef_[0][non_zero]
        scores = np.abs(contributions)
        terms = [
            (feature_names[index], float(contribution))
            for index, contribution in zip(non_zero, contributions)
        ]
        ordered = [term for _, term in sorted(zip(scores, terms), reverse=True)]
        return ordered[:top_k]

    if hasattr(model, "feature_log_prob_"):
        delta = model.feature_log_prob_[1][non_zero] - model.feature_log_prob_[0][non_zero]
        contributions = row[non_zero] * delta
        scores = np.abs(contributions)
        terms = [
            (feature_names[index], float(contribution))
            for index, contribution in zip(non_zero, contributions)
        ]
        ordered = [term for _, term in sorted(zip(scores, terms), reverse=True)]
        return ordered[:top_k]

    if hasattr(model, "feature_importances_"):
        contributions = row[non_zero] * model.feature_importances_[non_zero]
        terms = [
            (feature_names[index], float(contribution))
            for index, contribution in zip(non_zero, contributions)
        ]
        ordered = sorted(terms, key=lambda item: item[1], reverse=True)
        return ordered[:top_k]

    return []


def build_explanation(predicted_label: str, terms: List[Tuple[str, float]]) -> str:
    if not terms:
        return "The text is short or too generic, so the model could not identify strong clue words."

    highlight_words = ", ".join(term for term, _ in terms[:4])
    if predicted_label == "FAKE":
        return (
            f"The model leans toward FAKE because words such as {highlight_words} "
            "match patterns that appeared more often in misleading or sensational examples."
        )

    return (
        f"The model leans toward REAL because words such as {highlight_words} "
        "look more similar to informative language seen in trustworthy samples."
    )
