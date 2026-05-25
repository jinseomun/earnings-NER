import os
import json
import pandas as pd
import spacy
from datasets import load_dataset

from src.analysis.statistics import (
    clean_text,
    split_sentences_spacy,
    compute_basic_statistics,
    statistics_to_dataframe
)
from src.analysis.ngram import extract_ngrams
from src.analysis.tfidf import extract_tfidf_keywords


OUTPUT_DIR = "data/outputs/eda"
DATASET_NAME = "jlh-ibm/earnings_call"
DATASET_CONFIG = "transcripts"
SAMPLE_SIZE = None


def find_text_column(dataset_split):
    candidate_columns = [
        "transcript",
        "transcripts",
        "text",
        "content",
        "call_transcript",
        "transcript_text"
    ]

    for col in candidate_columns:
        if col in dataset_split.column_names:
            return col

    raise ValueError(
        f"Transcript column not found. Available columns: {dataset_split.column_names}"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading transcripts subset...")
    data = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train"
    )

    print(data)
    print("Columns:", data.column_names)

    text_col = find_text_column(data)
    print(f"Using text column: {text_col}")

    texts = [clean_text(t) for t in data[text_col] if clean_text(t)]

    if SAMPLE_SIZE is not None:
        texts = texts[:SAMPLE_SIZE]

    print(f"Number of transcripts used: {len(texts)}")

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Splitting sentences...")
    sentence_results = split_sentences_spacy(texts, nlp)

    print("Computing statistics...")
    stats = compute_basic_statistics(texts, sentence_results)
    stats_df = statistics_to_dataframe(stats)

    print("Extracting n-grams...")
    unigram_df = extract_ngrams(texts, ngram_range=(1, 1), max_features=50)
    bigram_df = extract_ngrams(texts, ngram_range=(2, 2), max_features=50)
    trigram_df = extract_ngrams(texts, ngram_range=(3, 3), max_features=50)

    print("Extracting TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(texts, max_features=50)

    stats_df.to_csv(f"{OUTPUT_DIR}/basic_statistics.csv", index=False)
    unigram_df.to_csv(f"{OUTPUT_DIR}/unigrams.csv", index=False)
    bigram_df.to_csv(f"{OUTPUT_DIR}/bigrams.csv", index=False)
    trigram_df.to_csv(f"{OUTPUT_DIR}/trigrams.csv", index=False)
    tfidf_df.to_csv(f"{OUTPUT_DIR}/tfidf_keywords.csv", index=False)

    with open(f"{OUTPUT_DIR}/sentence_results_sample.json", "w", encoding="utf-8") as f:
        json.dump(sentence_results[:20], f, ensure_ascii=False, indent=2)

    print("\n=== Basic Statistics ===")
    print(stats_df)

    print("\n=== Top Bigrams ===")
    print(bigram_df.head(20))

    print("\n=== Top TF-IDF Keywords ===")
    print(tfidf_df.head(20))

    print(f"\nSaved EDA outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()