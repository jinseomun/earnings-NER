import os
import json
import pandas as pd
import spacy

from src.analysis.ngram import extract_ngrams
from src.analysis.tfidf import extract_tfidf_keywords
from src.analysis.visualization import save_bar_chart


INPUT_SENTENCE_PATH = "data/outputs/preprocessing/sentence_split.json"
INPUT_SECTION_PATH = "data/outputs/preprocessing/section_split.json"
INPUT_SPEAKER_PATH = "data/outputs/preprocessing/speaker_split.json"

OUTPUT_EDA_DIR = "data/outputs/eda"
OUTPUT_FIGURE_DIR = "data/outputs/eda/figures"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_EDA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_EDA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

    print("Loading saved preprocessing outputs...")
    sentence_results = load_json(INPUT_SENTENCE_PATH)
    section_results = load_json(INPUT_SECTION_PATH)
    speaker_results = load_json(INPUT_SPEAKER_PATH)

    print(f"Loaded sentence utterances: {len(sentence_results)}")
    print(f"Loaded section items: {len(section_results)}")
    print(f"Loaded speaker items: {len(speaker_results)}")

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    utterance_texts = [
        " ".join(item["sentences"])
        for item in sentence_results
        if item.get("sentences")
    ]

    print(f"Using {len(utterance_texts)} utterances for keyword analysis")

    sentence_df = pd.DataFrame(sentence_results)
    section_df = pd.DataFrame(section_results)
    speaker_df = pd.DataFrame(speaker_results)

    print("Computing saved-data statistics...")

    basic_stats_df = pd.DataFrame([
        {"metric": "num_section_items", "value": len(section_results)},
        {"metric": "num_speaker_utterances", "value": len(speaker_results)},
        {"metric": "num_sentence_utterances", "value": len(sentence_results)},
        {"metric": "num_utterances_used_for_eda", "value": len(utterance_texts)},
        {"metric": "avg_sentences_per_utterance", "value": sentence_df["num_sentences"].mean()},
        {"metric": "max_sentences_per_utterance", "value": sentence_df["num_sentences"].max()},
        {"metric": "min_sentences_per_utterance", "value": sentence_df["num_sentences"].min()},
    ])

    section_stats_df = (
        sentence_df
        .groupby("section")
        .agg(
            num_utterances=("text", "count"),
            total_sentences=("num_sentences", "sum"),
            avg_sentences_per_utterance=("num_sentences", "mean")
        )
        .reset_index()
        .sort_values("num_utterances", ascending=False)
    )

    speaker_stats_df = (
        sentence_df
        .groupby("speaker")
        .agg(
            num_utterances=("text", "count"),
            total_sentences=("num_sentences", "sum"),
            avg_sentences_per_utterance=("num_sentences", "mean")
        )
        .reset_index()
        .sort_values("num_utterances", ascending=False)
    )

    print("Extracting n-grams...")
    unigram_df = extract_ngrams(utterance_texts, nlp, ngram_range=(1, 1), max_features=50)
    bigram_df = extract_ngrams(utterance_texts, nlp, ngram_range=(2, 2), max_features=50)
    trigram_df = extract_ngrams(utterance_texts, nlp, ngram_range=(3, 3), max_features=50)

    print("Extracting TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(utterance_texts, nlp, max_features=50)

    save_csv(basic_stats_df, "basic_statistics_from_preprocessed.csv")
    save_csv(section_stats_df, "section_statistics.csv")
    save_csv(speaker_stats_df, "speaker_statistics.csv")
    save_csv(unigram_df, "unigrams.csv")
    save_csv(bigram_df, "bigrams.csv")
    save_csv(trigram_df, "trigrams.csv")
    save_csv(tfidf_df, "tfidf_keywords.csv")

    print("Saving visualizations...")

    save_bar_chart(
        tfidf_df,
        x_col="term",
        y_col="tfidf_score",
        title="Top TF-IDF Keywords",
        output_path=f"{OUTPUT_FIGURE_DIR}/top_tfidf_keywords.png",
        top_n=20,
        xlabel="TF-IDF Score",
        ylabel="Term"
    )

    save_bar_chart(
        bigram_df,
        x_col="ngram",
        y_col="frequency",
        title="Top Bigrams",
        output_path=f"{OUTPUT_FIGURE_DIR}/top_bigrams.png",
        top_n=20,
        xlabel="Frequency",
        ylabel="Bigram"
    )

    save_bar_chart(
        trigram_df,
        x_col="ngram",
        y_col="frequency",
        title="Top Trigrams",
        output_path=f"{OUTPUT_FIGURE_DIR}/top_trigrams.png",
        top_n=20,
        xlabel="Frequency",
        ylabel="Trigram"
    )

    save_bar_chart(
        section_stats_df,
        x_col="section",
        y_col="num_utterances",
        title="Number of Utterances by Section",
        output_path=f"{OUTPUT_FIGURE_DIR}/section_utterance_counts.png",
        top_n=20,
        xlabel="Number of Utterances",
        ylabel="Section"
    )

    save_bar_chart(
        speaker_stats_df,
        x_col="speaker",
        y_col="num_utterances",
        title="Top Speakers by Number of Utterances",
        output_path=f"{OUTPUT_FIGURE_DIR}/speaker_utterance_counts.png",
        top_n=20,
        xlabel="Number of Utterances",
        ylabel="Speaker"
    )

    print("\n=== Basic Statistics ===")
    print(basic_stats_df)

    print("\n=== Section Statistics ===")
    print(section_stats_df)

    print("\n=== Speaker Statistics ===")
    print(speaker_stats_df.head(20))

    print("\n=== Top Unigrams ===")
    print(unigram_df.head(20))

    print("\n=== Top Bigrams ===")
    print(bigram_df.head(20))

    print("\n=== Top Trigrams ===")
    print(trigram_df.head(20))

    print("\n=== Top TF-IDF Keywords ===")
    print(tfidf_df.head(20))

    print("\nSaved EDA outputs to:", OUTPUT_EDA_DIR)
    print("Saved figures to:", OUTPUT_FIGURE_DIR)


if __name__ == "__main__":
    main()