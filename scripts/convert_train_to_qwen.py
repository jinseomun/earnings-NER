import os
import re
import json
from glob import glob


TXT_DIR = "data/annotation/teamtat_input"
LABEL_DIR = "data/annotation/train_labels"
OUTPUT_PATH = "data/annotation/qwen_train/qwen_train.jsonl"


INSTRUCTION = (
    "Extract readable business semantic spans from the earnings call utterance. "
    "Use only these labels: PRODUCT, SEGMENT, GROWTH_SIGNAL. "
    "Return a JSON array of objects with label and text. "
    "Do not add explanations."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_utterances(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"\[UTTERANCE_ID=(\d+)\]\n(.*?)(?=\n\[UTTERANCE_BOUNDARY\])",
        re.DOTALL
    )

    utterances = {}
    for match in pattern.finditer(text):
        utterance_id = int(match.group(1))
        utterance_text = match.group(2).strip()
        utterances[utterance_id] = utterance_text

    return utterances


def build_record(utterance_text, annotations):
    return {
        "messages": [
            {
                "role": "system",
                "content": INSTRUCTION
            },
            {
                "role": "user",
                "content": utterance_text
            },
            {
                "role": "assistant",
                "content": json.dumps(annotations, ensure_ascii=False)
            }
        ]
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    train_files = sorted(glob(os.path.join(LABEL_DIR, "*_train.json")))

    count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for label_path in train_files:
            base_name = os.path.basename(label_path).replace("_train.json", "")
            txt_path = os.path.join(TXT_DIR, f"{base_name}.txt")

            if not os.path.exists(txt_path):
                print(f"Missing txt file: {txt_path}")
                continue

            utterances = parse_utterances(txt_path)
            labels = load_json(label_path)

            for item in labels:
                utterance_id = item["utterance_id"]
                annotations = item.get("annotations", [])

                if utterance_id not in utterances:
                    print(f"Missing utterance_id={utterance_id} in {txt_path}")
                    continue

                record = build_record(
                    utterance_text=utterances[utterance_id],
                    annotations=annotations
                )

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()