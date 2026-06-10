import json
import random
import re
from pathlib import Path


LABEL_DIR = Path("data/annotation/gpt_relabel")
INPUT_DIR = Path("data/annotation/teamtat_input")
OUT_DIR = Path("data/annotation/splits")

SEED = 42
TRAIN_SIZE = 140
VALID_SIZE = 30
TEST_SIZE = 30


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_teamtat_input_text(text):
    """
    TeamTAT input txt format:
    [UTTERANCE_ID=0]
    ...
    [UTTERANCE_BOUNDARY]
    """
    pattern = re.compile(
        r"\[UTTERANCE_ID=(\d+)\]\s*(.*?)\s*\[UTTERANCE_BOUNDARY\]",
        re.DOTALL,
    )

    text_map = {}

    for match in pattern.finditer(text):
        utterance_id = int(match.group(1))
        utterance_text = match.group(2).strip()
        text_map[utterance_id] = utterance_text

    return text_map


def load_all_texts():
    text_map = {}

    files = sorted(INPUT_DIR.glob("teamtat_doc_*_gpt.txt"))

    if not files:
        files = sorted(INPUT_DIR.glob("*.txt"))

    for file in files:
        raw_text = file.read_text(encoding="utf-8")
        parsed = parse_teamtat_input_text(raw_text)
        text_map.update(parsed)

    return text_map


def load_all_gold():
    all_items = []

    files = sorted(LABEL_DIR.glob("teamtat_doc_*_gpt.json"))

    for file in files:
        data = load_json(file)

        for item in data:
            all_items.append({
                "utterance_id": item["utterance_id"],
                "annotations": item.get("annotations", []),
                "source_file": file.name,
            })

    return all_items


def merge_gold_with_text(gold_items, text_map):
    merged = []
    missing = []

    for item in gold_items:
        uid = item["utterance_id"]

        if uid not in text_map:
            missing.append(uid)
            continue

        merged.append({
            "utterance_id": uid,
            "text": text_map[uid],
            "annotations": item.get("annotations", []),
            "source_file": item.get("source_file", ""),
        })

    if missing:
        print(f"[WARN] Missing text for {len(missing)} utterances:")
        print(missing[:20])

    return merged


def split_data(data):
    random.seed(SEED)
    data = data[:]
    random.shuffle(data)

    train = data[:TRAIN_SIZE]
    valid = data[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]
    test = data[TRAIN_SIZE + VALID_SIZE:TRAIN_SIZE + VALID_SIZE + TEST_SIZE]

    return train, valid, test


def label_counts(data):
    counts = {
        "PRODUCT": 0,
        "SEGMENT": 0,
        "GROWTH_SIGNAL": 0,
    }

    for item in data:
        for ann in item.get("annotations", []):
            label = ann.get("label")
            if label in counts:
                counts[label] += 1

    return counts


def print_stats(name, data):
    total_entities = sum(len(item.get("annotations", [])) for item in data)
    annotated = sum(1 for item in data if item.get("annotations"))

    print(f"\n=== {name} ===")
    print(f"Utterances: {len(data)}")
    print(f"Annotated utterances: {annotated}")
    print(f"Empty utterances: {len(data) - annotated}")
    print(f"Total entities: {total_entities}")
    print(f"Label counts: {label_counts(data)}")


def main():
    text_map = load_all_texts()
    gold_items = load_all_gold()

    print(f"Loaded texts: {len(text_map)}")
    print(f"Loaded gold items: {len(gold_items)}")

    merged = merge_gold_with_text(gold_items, text_map)

    print(f"Merged items: {len(merged)}")

    if len(merged) < TRAIN_SIZE + VALID_SIZE + TEST_SIZE:
        raise ValueError(
            f"Not enough merged data: {len(merged)} "
            f"< {TRAIN_SIZE + VALID_SIZE + TEST_SIZE}"
        )

    train, valid, test = split_data(merged)

    save_json(train, OUT_DIR / "train.json")
    save_json(valid, OUT_DIR / "valid.json")
    save_json(test, OUT_DIR / "test.json")

    print_stats("Train", train)
    print_stats("Valid", valid)
    print_stats("Test", test)

    print("\nSaved:")
    print(OUT_DIR / "train.json")
    print(OUT_DIR / "valid.json")
    print(OUT_DIR / "test.json")


if __name__ == "__main__":
    main()