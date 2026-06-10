import json
from pathlib import Path
from collections import Counter


LABEL_DIR = Path("data/annotation/gpt_relabel")
VALID_LABELS = {"PRODUCT", "SEGMENT", "GROWTH_SIGNAL"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    files = sorted(LABEL_DIR.glob("teamtat_doc_*_gpt.json"))

    utterance_count = 0
    annotated_utterance_count = 0
    total_entities = 0

    label_counter = Counter()
    empty_utterance_count = 0

    for file in files:
        data = load_json(file)

        for item in data:
            utterance_count += 1
            annotations = item.get("annotations", [])

            if annotations:
                annotated_utterance_count += 1
            else:
                empty_utterance_count += 1

            for ann in annotations:
                label = ann.get("label")
                text = ann.get("text", "").strip()

                if label not in VALID_LABELS:
                    print(f"[WARN] invalid label in {file.name}: {label}")

                if not text:
                    print(f"[WARN] empty text in {file.name}: {item.get('utterance_id')}")

                label_counter[label] += 1
                total_entities += 1

    avg_entities_per_utterance = (
        total_entities / utterance_count if utterance_count else 0
    )

    avg_entities_per_annotated_utterance = (
        total_entities / annotated_utterance_count
        if annotated_utterance_count
        else 0
    )

    print("=== Dataset Statistics ===")
    print(f"Files: {len(files)}")
    print(f"Total utterances: {utterance_count}")
    print(f"Annotated utterances: {annotated_utterance_count}")
    print(f"Empty utterances: {empty_utterance_count}")
    print(f"Total entities: {total_entities}")
    print(f"Avg entities / utterance: {avg_entities_per_utterance:.2f}")
    print(
        "Avg entities / annotated utterance: "
        f"{avg_entities_per_annotated_utterance:.2f}"
    )

    print("\n=== Label Counts ===")
    for label in sorted(VALID_LABELS):
        print(f"{label}: {label_counter[label]}")

    print("\n=== LaTeX table values ===")
    print(f"Utterances & {utterance_count} \\\\")
    print(f"Annotated utterances & {annotated_utterance_count} \\\\")
    print(f"Total entities & {total_entities} \\\\")
    print(f"PRODUCT & {label_counter['PRODUCT']} \\\\")
    print(f"SEGMENT & {label_counter['SEGMENT']} \\\\")
    print(f"GROWTH\\_SIGNAL & {label_counter['GROWTH_SIGNAL']} \\\\")


if __name__ == "__main__":
    main()