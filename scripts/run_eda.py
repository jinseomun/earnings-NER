import os
import json
import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

from src.analysis.statistics import (
    compute_basic_statistics,
    statistics_to_dataframe
)
from src.analysis.ngram import extract_ngrams
from src.analysis.tfidf import extract_tfidf_keywords

from src.preprocessing.text_cleaner import clean_transcript
from src.preprocessing.section_split import split_sections
from src.preprocessing.speaker_split import split_speakers
from src.preprocessing.sentence_split import split_sentences


OUTPUT_PREPROCESSING_DIR = "data/outputs/preprocessing"
OUTPUT_EDA_DIR = "data/outputs/eda"

DATASET_NAME = "jlh-ibm/earnings_call"
DATASET_CONFIG = "transcripts"

SAMPLE_SIZE = 40


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


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_PREPROCESSING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_EDA_DIR, exist_ok=True)

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

    texts = []

    for t in data[text_col]:
        cleaned = clean_transcript(t)

        if cleaned:
            texts.append(cleaned)

    if SAMPLE_SIZE is not None:
        texts = texts[:SAMPLE_SIZE]

    print(f"Number of transcripts used: {len(texts)}")

    print("Section splitting...")
    section_results = []

    for doc_id, text in enumerate(tqdm(texts, desc="Section Splitting")):
        sections = split_sections(text)

        for section in sections:
            section_results.append({
                "doc_id": doc_id,
                "section": section["section"],
                "text": section["text"]
            })

    save_json(
        section_results,
        f"{OUTPUT_PREPROCESSING_DIR}/section_split.json"
    )

    print("Speaker splitting...")
    speaker_results = []

    for doc_id in tqdm(range(len(texts)), desc="Speaker Splitting"):
        doc_sections = [
            item for item in section_results
            if item["doc_id"] == doc_id
        ]

        speaker_items = split_speakers(doc_sections)

        for item in speaker_items:
            speaker_results.append({
                "doc_id": doc_id,
                "section": item["section"],
                "speaker": item["speaker"],
                "text": item["text"]
            })

    save_json(
        speaker_results,
        f"{OUTPUT_PREPROCESSING_DIR}/speaker_split.json"
    )

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Sentence splitting...")
    sentence_results = split_sentences(speaker_results, nlp)

    save_json(
        sentence_results,
        f"{OUTPUT_PREPROCESSING_DIR}/sentence_split.json"
    )

    # sentence filtering까지 끝난 문장만 TF-IDF / n-gram 입력으로 사용
    utterance_texts = [
        " ".join(item["sentences"])
        for item in sentence_results
        if item["sentences"]
    ]

    print("Computing basic statistics...")
    stats = compute_basic_statistics(texts)
    stats_df = statistics_to_dataframe(stats)

    section_df = pd.DataFrame(sentence_results)

    section_stats_df = (
        section_df
        .groupby("section")
        .agg(
            num_utterances=("text", "count"),
            total_sentences=("num_sentences", "sum"),
            avg_sentences_per_utterance=("num_sentences", "mean")
        )
        .reset_index()
    )

    speaker_stats_df = (
        section_df
        .groupby("speaker")
        .agg(
            num_utterances=("text", "count"),
            total_sentences=("num_sentences", "sum"),
            avg_sentences_per_utterance=("num_sentences", "mean")
        )
        .sort_values("num_utterances", ascending=False)
        .reset_index()
    )

    print("Extracting n-grams...")
    unigram_df = extract_ngrams(
        utterance_texts,
        nlp,
        ngram_range=(1, 1),
        max_features=50
    )

    bigram_df = extract_ngrams(
        utterance_texts,
        nlp,
        ngram_range=(2, 2),
        max_features=50
    )

    trigram_df = extract_ngrams(
        utterance_texts,
        nlp,
        ngram_range=(3, 3),
        max_features=50
    )

    print("Extracting TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(
        utterance_texts,
        nlp,
        max_features=50
    )

    stats_df.to_csv(f"{OUTPUT_EDA_DIR}/basic_statistics.csv", index=False)
    section_stats_df.to_csv(f"{OUTPUT_EDA_DIR}/section_statistics.csv", index=False)
    speaker_stats_df.to_csv(f"{OUTPUT_EDA_DIR}/speaker_statistics.csv", index=False)

    unigram_df.to_csv(f"{OUTPUT_EDA_DIR}/unigrams.csv", index=False)
    bigram_df.to_csv(f"{OUTPUT_EDA_DIR}/bigrams.csv", index=False)
    trigram_df.to_csv(f"{OUTPUT_EDA_DIR}/trigrams.csv", index=False)
    tfidf_df.to_csv(f"{OUTPUT_EDA_DIR}/tfidf_keywords.csv", index=False)

    print("\n=== Basic Statistics ===")
    print(stats_df)

    print("\n=== Section Statistics ===")
    print(section_stats_df)

    print("\n=== Speaker Statistics ===")
    print(speaker_stats_df.head(20))

    print("\n=== Top Bigrams ===")
    print(bigram_df.head(20))

    print("\n=== Top TF-IDF Keywords ===")
    print(tfidf_df.head(20))

    print("\nSaved preprocessing outputs to:", OUTPUT_PREPROCESSING_DIR)
    print("Saved EDA outputs to:", OUTPUT_EDA_DIR)


if __name__ == "__main__":
    main()