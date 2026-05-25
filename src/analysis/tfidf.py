import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CUSTOM_STOPWORDS = {
    "question", "questions", "answer", "answers",
    "thank", "thanks", "okay", "yes", "no",
    "great", "sure", "right",
    "think", "just", "like", "going", "look", "really",
    "today", "bit", "maybe", "let", "want",
    "operator", "please"
}


def build_spacy_tokenizer(nlp):
    def spacy_tokenizer(text):
        doc = nlp(text)

        tokens = []

        for token in doc:
            lemma = token.lemma_.lower().strip()

            if token.is_stop:
                continue

            if token.is_punct or token.is_space:
                continue

            if not lemma.isalpha():
                continue

            if lemma in CUSTOM_STOPWORDS:
                continue

            if len(lemma) <= 2:
                continue

            tokens.append(lemma)

        return tokens

    return spacy_tokenizer


def extract_tfidf_keywords(
    texts,
    nlp,
    max_features=50,
    min_df=2
):
    tokenizer = build_spacy_tokenizer(nlp)

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        lowercase=False,
        token_pattern=None,
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