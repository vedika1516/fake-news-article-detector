from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_detector.explainability import (  # noqa: E402
    build_explanation,
    important_terms,
    predict_with_details,
)
from fake_news_detector.modeling import load_model_bundle  # noqa: E402
from fake_news_detector.scraper import fetch_article_text  # noqa: E402
from fake_news_detector.text_processing import preprocess_text  # noqa: E402


ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.pkl"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics_summary.csv"

SAMPLE_NEWS = {
    "Fake style sample": (
        "Miracle herb discovered to cure every disease overnight, doctors furious as secret government report leaks."
    ),
    "Real style sample": (
        "The state health department released its weekly report showing a modest decline in seasonal flu cases."
    ),
    "Political sample": (
        "Officials said the bill passed committee after a two-hour debate and will move to the assembly floor next week."
    ),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(13, 24, 40, 0.92);
            --panel-2: rgba(17, 33, 54, 0.92);
            --accent: #4fd1c5;
            --accent-2: #f6ad55;
            --text: #e8f1ff;
            --muted: #9fb6d3;
            --danger: #ff6b6b;
            --success: #6ee7b7;
            --border: rgba(159, 182, 211, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 209, 197, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(246, 173, 85, 0.14), transparent 24%),
                linear-gradient(180deg, #02060d 0%, #07111f 45%, #0b1727 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(79, 209, 197, 0.16), rgba(246, 173, 85, 0.14));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.3rem;
            letter-spacing: 0.03em;
            color: var(--text);
        }

        .muted {
            color: var(--muted);
        }

        .meter-wrap {
            background: rgba(255,255,255,0.06);
            border-radius: 999px;
            overflow: hidden;
            height: 18px;
            margin-top: 0.4rem;
        }

        .meter-fill {
            height: 18px;
            background: linear-gradient(90deg, #ff6b6b 0%, #f6ad55 45%, #4fd1c5 100%);
            border-radius: 999px;
        }

        .pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border: 1px solid var(--border);
            background: var(--panel-2);
            font-size: 0.92rem;
        }

        .fake {
            color: #ffd5d5;
            border-color: rgba(255, 107, 107, 0.4);
        }

        .real {
            color: #d6fff0;
            border-color: rgba(110, 231, 183, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_bundle():
    return load_model_bundle(ARTIFACT_PATH)


@st.cache_data
def get_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


def render_meter(prob_real: float) -> None:
    width = max(1, min(int(prob_real * 100), 100))
    st.markdown(
        f"""
        <div class="panel">
            <strong>Real vs Fake Probability Meter</strong>
            <div class="muted">Left means more FAKE, right means more REAL.</div>
            <div class="meter-wrap">
                <div class="meter-fill" style="width: {width}%;"></div>
            </div>
            <div class="muted" style="margin-top:0.45rem;">REAL probability: {prob_real * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_terms(terms, predicted_label: str) -> None:
    if not terms:
        st.info("No strong clue words were detected for this text.")
        return

    css_class = "fake" if predicted_label == "FAKE" else "real"
    html = "".join(
        f'<span class="pill {css_class}">{term} ({abs(score):.3f})</span>'
        for term, score in terms
    )
    st.markdown(f'<div class="panel"><strong>Important Words</strong><br>{html}</div>', unsafe_allow_html=True)


def append_history(result: dict, text: str) -> None:
    history = st.session_state.setdefault("history", [])
    history.insert(
        0,
        {
            "prediction": result["predicted_label"],
            "confidence": round(result["confidence"] * 100, 2),
            "model": result["model_name"],
            "preview": text[:120].replace("\n", " "),
        },
    )
    st.session_state["history"] = history[:10]


def main() -> None:
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    inject_css()

    if not ARTIFACT_PATH.exists():
        st.error(
            "Model artifacts are missing. Run `python3 train.py --data data/demo_fake_news_dataset.csv --output-dir artifacts` first."
        )
        st.stop()

    bundle = get_bundle()
    metrics_df = get_metrics()
    model_names = list(bundle["models"].keys())

    with st.sidebar:
        st.header("Settings")
        default_index = model_names.index(bundle["best_model_name"])
        selected_model = st.selectbox("Choose model", model_names, index=default_index)

        st.markdown("### About Project")
        st.write(
            "This app classifies news text as REAL or FAKE using TF-IDF features and multiple machine learning models."
        )
        st.write(
            "It compares Logistic Regression, Naive Bayes, and Random Forest, then highlights clue words behind each prediction."
        )
        st.markdown("### Metrics Snapshot")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="hero">
            <h1>Fake News Detection Studio</h1>
            <div class="muted">Analyze article text, estimate confidence, inspect important words, and even pull content directly from a URL.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_columns = st.columns(3)
    for index, (label, sample_text) in enumerate(SAMPLE_NEWS.items()):
        if sample_columns[index].button(label, use_container_width=True):
            st.session_state["news_input"] = sample_text

    url_col, scrape_col = st.columns([4, 1])
    with url_col:
        url_value = st.text_input("Paste article URL (optional)", placeholder="https://example.com/news-story")
    with scrape_col:
        scrape_clicked = st.button("Extract URL", use_container_width=True)

    if scrape_clicked and url_value:
        try:
            title, article_text = fetch_article_text(url_value)
            st.session_state["news_input"] = f"{title}. {article_text}"
            st.success("Article content extracted successfully.")
        except Exception as exc:
            st.error(f"URL extraction failed: {exc}")

    news_text = st.text_area(
        "Paste news text here",
        key="news_input",
        height=220,
        placeholder="Enter a headline, a paragraph, or a full article...",
    )

    predict_clicked = st.button("Predict News Type", type="primary", use_container_width=True)

    if predict_clicked:
        cleaned = preprocess_text(news_text)
        if not cleaned:
            st.warning("Please enter some meaningful text before predicting.")
        else:
            result = predict_with_details(bundle, cleaned, model_name=selected_model)
            terms = important_terms(bundle, selected_model, result["features"])
            explanation = build_explanation(result["predicted_label"], terms)
            append_history(result, news_text)

            pred_col, conf_col, model_col = st.columns(3)
            pred_col.metric("Prediction", result["predicted_label"])
            conf_col.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
            model_col.metric("Model", result["model_name"])

            probs_df = pd.DataFrame(
                {
                    "Label": ["FAKE", "REAL"],
                    "Probability": [result["prob_fake"], result["prob_real"]],
                }
            ).set_index("Label")

            chart_col, info_col = st.columns([1, 1])
            with chart_col:
                st.markdown('<div class="panel"><strong>Probability Chart</strong></div>', unsafe_allow_html=True)
                st.bar_chart(probs_df)
                render_meter(result["prob_real"])

            with info_col:
                st.markdown('<div class="panel"><strong>Model Explanation</strong></div>', unsafe_allow_html=True)
                st.write(explanation)
                render_terms(terms, result["predicted_label"])

    history = st.session_state.get("history", [])
    if history:
        st.markdown("### Recent Prediction History")
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
