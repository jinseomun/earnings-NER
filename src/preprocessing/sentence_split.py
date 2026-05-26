from tqdm import tqdm
from src.preprocessing.text_filter import filter_sentences


def split_sentences(items, nlp):
    results = []
    texts = [item["text"] for item in items]

    for item, doc in tqdm(
        zip(items, nlp.pipe(texts, batch_size=8)),
        total=len(items),
        desc="Sentence Splitting"
    ):
        raw_sentences = [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip()
        ]

        sentences = filter_sentences(raw_sentences)

        results.append({
            "doc_id": item.get("doc_id"),
            "section": item["section"],
            "speaker": item["speaker"],
            "text": item["text"],
            "num_sentences": len(sentences),
            "sentences": sentences
        })

    return results