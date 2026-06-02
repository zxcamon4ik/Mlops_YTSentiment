import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.common.text_preprocessing import TextPreprocessor


def build_sentiment_pipeline(
    max_features: int,
    ngram_range: tuple[int, int],
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
) -> Pipeline:
    classifier = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        is_unbalance=True,
        class_weight="balanced",
        reg_alpha=0.1,
        reg_lambda=0.1,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
    )

    return Pipeline(
        steps=[
            ("preprocess", TextPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            ),
            ("classifier", classifier),
        ]
    )
