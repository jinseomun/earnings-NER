import os
import re
import json


INPUT_TXT = "data/annotation/teamtat_input/teamtat_doc_011.txt"
OUTPUT_PATH = "data/annotation/eval/teamtat_doc_011_inputs.json"


def parse_utterances(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"\[UTTERANCE_ID=(\d+)\]\n(.*?)(?=\n\[UTTERANCE_BOUNDARY\])",
        re.DOTALL
    )

    data = []

    for match in pattern.finditer(text):
        data.append({
            "utterance_id": int(match.group(1)),
            "text": match.group(2).strip()
        })

    return data


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    data = parse_utterances(INPUT_TXT)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} test inputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()