from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_detector.data_utils import load_dataset
from fake_news_detector.modeling import save_training_artifacts, train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument(
        "--data",
        default="data/demo_fake_news_dataset.csv",
        help="Path to a dataset file or dataset folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where artifacts will be written.",
    )
    parser.add_argument(
        "--enable-word2vec",
        action="store_true",
        help="Train an optional Word2Vec-based experiment if gensim is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, source_name = load_dataset(args.data)
    artifacts = train_all_models(
        df,
        dataset_source=source_name,
        enable_word2vec=args.enable_word2vec,
    )
    bundle_path = save_training_artifacts(artifacts, args.output_dir)

    print(f"Dataset source: {source_name}")
    print("Model comparison:")
    print(artifacts.metrics.to_string(index=False))
    print(f"\nBest model selected automatically: {artifacts.best_model_name}")
    print(f"Artifacts saved to: {bundle_path}")


if __name__ == "__main__":
    main()
