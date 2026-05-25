import re
import pandas as pd
from tqdm import tqdm


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences_spacy(texts, nlp, limit=None):
    results = []

    if limit:
        texts = texts[:limit]

    for idx, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=8),
            total=len(texts),
            desc="Sentence Splitting"
        )
    ):
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        results.append({
            "doc_id": idx,
            "num_sentences": len(sentences),
            "sentences": sentences
        })

    return results


def compute_basic_statistics(texts, sentence_results=None):
    cleaned = [clean_text(t) for t in texts]

    word_counts = [len(t.split()) for t in cleaned]
    char_counts = [len(t) for t in cleaned]

    stats = {
        "num_transcripts": len(cleaned),
        "avg_word_count": sum(word_counts) / len(word_counts),
        "max_word_count": max(word_counts),
        "min_word_count": min(word_counts),
        "avg_char_count": sum(char_counts) / len(char_counts),
        "max_char_count": max(char_counts),
        "min_char_count": min(char_counts),
    }

    if sentence_results:
        sentence_counts = [r["num_sentences"] for r in sentence_results]
        stats.update({
            "avg_sentence_count": sum(sentence_counts) / len(sentence_counts),
            "max_sentence_count": max(sentence_counts),
            "min_sentence_count": min(sentence_counts),
        })

    return stats


def statistics_to_dataframe(stats: dict):
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in stats.items()]
    )