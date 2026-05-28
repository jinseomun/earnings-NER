import os
import json
import random


INPUT_PATH = "data/outputs/preprocessing/speaker_split.json"
OUTPUT_DIR = "data/annotation/teamtat_input"

TOTAL_UTTERANCES = 300
MAX_UTTERANCES_PER_FILE = 20
RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(INPUT_PATH)

    utterances = []

    for idx, item in enumerate(data):
        text = item.get("text", "").strip()

        if not text:
            continue

        utterances.append({
            "utterance_id": idx,
            "doc_id": item.get("doc_id"),
            "section": item.get("section", "unknown"),
            "speaker": item.get("speaker", "unknown"),
            "text": text
        })

    print(f"Total available utterances: {len(utterances)}")

    random.seed(RANDOM_SEED)
    random.shuffle(utterances)

    selected_utterances = utterances[:TOTAL_UTTERANCES]

    metadata = []
    file_count = 0

    for start in range(0, len(selected_utterances), MAX_UTTERANCES_PER_FILE):
        chunk = selected_utterances[start:start + MAX_UTTERANCES_PER_FILE]

        file_name = f"teamtat_doc_{file_count:03d}.txt"
        output_path = os.path.join(OUTPUT_DIR, file_name)

        with open(output_path, "w", encoding="utf-8") as f:
            for local_idx, item in enumerate(chunk):
                sample_id = start + local_idx

                f.write(f"[UTTERANCE_ID={sample_id}]\n")
                f.write(item["text"])
                f.write("\n\n[UTTERANCE_BOUNDARY]\n\n")

                metadata.append({
                    "sample_id": sample_id,
                    "utterance_id": item["utterance_id"],
                    "teamtat_file": file_name,
                    "doc_id": item["doc_id"],
                    "section": item["section"],
                    "speaker": item["speaker"],
                    "text": item["text"]
                })

        print(f"Saved: {output_path}")
        file_count += 1

    metadata_path = os.path.join(OUTPUT_DIR, "teamtat_metadata.json")
    save_json(metadata, metadata_path)

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved {file_count} TeamTat input files.")


if __name__ == "__main__":
    main()