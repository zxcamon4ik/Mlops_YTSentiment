import re
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


IMPORTANT_SENTIMENT_WORDS = {"not", "but", "however", "no", "yet"}
FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}


def download_nltk_resources(download_dir: str | None = None) -> None:
    """Download NLTK resources used by training and serving."""
    import nltk

    kwargs = {"quiet": True}
    if download_dir:
        kwargs["download_dir"] = download_dir

    nltk.download("stopwords", **kwargs)
    nltk.download("wordnet", **kwargs)
    nltk.download("omw-1.4", **kwargs)


@lru_cache(maxsize=1)
def sentiment_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english")) - IMPORTANT_SENTIMENT_WORDS
    except LookupError:
        return FALLBACK_STOPWORDS - IMPORTANT_SENTIMENT_WORDS


@lru_cache(maxsize=1)
def lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def preprocess_comment(comment: object) -> str:
    """Apply the text normalization used by DVC and the inference pipeline."""
    text = "" if comment is None else str(comment)
    text = text.lower().strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)
    text = " ".join(
        word for word in text.split() if word not in sentiment_stopwords()
    )

    try:
        text = " ".join(lemmatizer().lemmatize(word) for word in text.split())
    except LookupError:
        pass

    return text


def text_values(values: object) -> List[str]:
    """Normalize supported sklearn/MLflow input shapes into a text list."""
    if isinstance(values, pd.DataFrame):
        for column in ("comment", "text", "clean_comment"):
            if column in values.columns:
                return values[column].fillna("").astype(str).tolist()
        if len(values.columns) == 0:
            return []
        return values.iloc[:, 0].fillna("").astype(str).tolist()

    if isinstance(values, pd.Series):
        return values.fillna("").astype(str).tolist()

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return [str(values.item())]
        if values.ndim > 1:
            values = values[:, 0]
        return ["" if item is None else str(item) for item in values.tolist()]

    if isinstance(values, str):
        return [values]

    if isinstance(values, dict):
        for key in ("comments", "comment", "text", "clean_comment"):
            if key in values:
                return text_values(values[key])

    if isinstance(values, Iterable):
        result: List[str] = []
        for item in values:
            if isinstance(item, dict):
                result.append(str(item.get("comment", item.get("text", ""))))
            else:
                result.append("" if item is None else str(item))
        return result

    return [str(values)]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn transformer that prepares raw comments before vectorization."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_comment(comment) for comment in text_values(X)]
