import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def extract_ngrams(
    texts,
    ngram_range=(1, 2),
    max_features=50,
    min_df=2
):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df
    )

    X = vectorizer.fit_transform(texts)

    terms = vectorizer.get_feature_names_out()
    freqs = X.sum(axis=0).A1

    df = pd.DataFrame({
        "ngram": terms,
        "frequency": freqs
    })

    return df.sort_values("frequency", ascending=False).reset_index(drop=True)