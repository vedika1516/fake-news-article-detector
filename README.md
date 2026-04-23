# Fake News Detection with Streamlit

This repository is a GitHub-ready, beginner-friendly end-to-end Machine Learning project for classifying news as `REAL` or `FAKE`.

It includes:

- data preprocessing with lowercasing, punctuation removal, stopword removal, and stemming
- multiple models: Logistic Regression, Multinomial Naive Bayes, and Random Forest
- automatic best-model selection using validation metrics
- TF-IDF feature extraction
- optional advanced Word2Vec-based experiment when `gensim` is installed
- explainability with important word highlighting and model confidence
- URL-based article extraction
- a Streamlit app with dark mode, sample news buttons, history, and a probability meter

## GitHub Ready Highlights

- clean project structure for direct upload to GitHub
- `.gitignore` included to avoid committing local virtual environments and cache files
- `LICENSE` included
- trained demo artifacts already included so the app can run immediately after cloning
- beginner-friendly code organized into reusable modules

## Folder Structure

```text
fake_news_detector/
├── app.py
├── train.py
├── train_fake_news_model.py
├── requirements.txt
├── README.md
├── artifacts/
│   ├── metrics_summary.csv
│   ├── model_bundle.pkl
│   └── training_summary.json
├── data/
│   └── demo_fake_news_dataset.csv
└── src/
    └── fake_news_detector/
        ├── __init__.py
        ├── data_utils.py
        ├── explainability.py
        ├── modeling.py
        ├── scraper.py
        └── text_processing.py
```

## Dataset Options

The training script supports multiple public dataset formats:

1. Kaggle Fake and True News dataset as a folder containing `Fake.csv` and `True.csv`
2. A single CSV with `title`, `text`, and `label`
3. LIAR dataset TSV files, with automatic conversion to binary labels

This repository also includes `data/demo_fake_news_dataset.csv` so the project runs immediately without an external download.

## How to Run Locally

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models:

```bash
python3 train.py --data "data/demo_fake_news_dataset.csv" --output-dir artifacts
```

You can also train on a Kaggle dataset folder:

```bash
python3 train.py --data "/path/to/fake-and-true-news" --output-dir artifacts
```

4. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Quick GitHub Upload Steps

1. Open a terminal in this project folder.
2. Initialize git:

```bash
git init
git add .
git commit -m "Initial commit - fake news detection project"
```

3. Create a new empty GitHub repository.
4. Connect your local folder to GitHub:

```bash
git remote add origin https://github.com/your-username/fake-news-detector.git
git branch -M main
git push -u origin main
```

## Streamlit Cloud Deployment

1. Push this folder to a GitHub repository.
2. Make sure `requirements.txt` and `app.py` are in the project root.
3. Open Streamlit Community Cloud.
4. Click `New app`.
5. Select your repo, branch, and set the main file path to `app.py`.
6. Deploy.

If the app opens before artifacts exist, train locally once and commit the generated `artifacts/model_bundle.pkl`, or add a startup step in your deployment workflow that runs `train.py`.

## Notes for Evaluation

- The demo dataset is intentionally small so the project is easy to run in a college environment.
- For better real-world performance, replace it with a larger public dataset such as the Kaggle Fake and True News dataset or LIAR.
- Word2Vec is optional and only runs when `gensim` is installed and the flag `--enable-word2vec` is used.
- The included `artifacts/model_bundle.pkl` makes the repository heavier, but it helps evaluators run the app instantly without training first.
