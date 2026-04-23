from __future__ import annotations

from typing import Tuple


def fetch_article_text(url: str) -> Tuple[str, str]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional at runtime
        raise RuntimeError(
            "requests and beautifulsoup4 are required for URL extraction."
        ) from exc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled article"

    paragraphs = []
    article = soup.find("article")
    candidates = article.find_all("p") if article else soup.find_all("p")
    for paragraph in candidates:
        text = paragraph.get_text(" ", strip=True)
        if len(text) >= 60:
            paragraphs.append(text)

    article_text = " ".join(paragraphs[:25]).strip()
    if not article_text:
        raise ValueError("Could not extract enough article text from the URL.")

    return title, article_text
