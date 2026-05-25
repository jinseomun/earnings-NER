import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.analysis.tfidf import build_spacy_tokenizer


def extract_ngrams(
    texts,
    nlp,
    ngram_range=(1, 2),
    max_features=50,
    min_df=2
):
    tokenizer = build_spacy_tokenizer(nlp)

    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        lowercase=False,
        token_pattern=None,
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