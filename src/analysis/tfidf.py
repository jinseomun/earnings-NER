import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_keywords(
    texts,
    max_features=50,
    min_df=2
):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df
    )

    X = vectorizer.fit_transform(texts)

    terms = vectorizer.get_feature_names_out()
    scores = X.mean(axis=0).A1

    df = pd.DataFrame({
        "term": terms,
        "tfidf_score": scores
    })

    return df.sort_values("tfidf_score", ascending=False).reset_index(drop=True)