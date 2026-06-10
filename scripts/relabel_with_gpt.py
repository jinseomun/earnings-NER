import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


INPUT_DIR = Path("data/annotation/teamtat_input")
LABEL_DIR = Path("data/annotation/train_labels")
OUT_DIR = Path("data/annotation/gpt_relabel")

MODEL = "gpt-4.1-mini"
SLEEP_SECONDS = 0.5

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are an expert annotator for Earnings Call Transcript information extraction.

Revise entity annotations according to the guideline.

Entity types:

1. PRODUCT
- Products, services, platforms, software, hardware, technologies, or product lines.
- Examples: Azure, Office 365, Microsoft 365, Ryzen, EPYC, Radeon, DGX, A100, Prime Video, Windows 10, 3D NAND.

2. SEGMENT
- Business divisions, business units, customer segments, market segments, or reporting segments.
- Examples: Data Center Group, Client Computing Group, Computing and Graphics segment,
  Enterprise, Embedded and Semi-Custom segment, Internet of Things, Memory business,
  Security business, Infrastructure Platforms, Applications, Enterprise, Commercial,
  Public sector, Service Provider, PC segment, Cloud segment.

3. GROWTH_SIGNAL
- A semantically complete span expressing growth, decline, demand change, performance change,
  financial result, outlook, guidance, momentum, constraint, or strategic progress.
- Good examples:
  "revenue increased 20% sequentially"
  "strong demand"
  "growth rate to accelerate"
  "gross margin compression"
  "expect revenue growth in the mid single digits"
  "chip supply to remain constrained"

Rules:
- Keep spans as exact substrings from the utterance.
- Do not invent text that does not appear in the utterance.
- Remove incorrect or overly vague annotations.
- Add missing PRODUCT, SEGMENT, and GROWTH_SIGNAL annotations.
- Do not label company names as PRODUCT.
- Do not label accounting standards, acquisitions, or legal events as PRODUCT unless they are clearly a product/service/platform.
- Return JSON only.
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_teamtat_input_text(text):
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


def load_text_map():
    text_map = {}

    for file in sorted(INPUT_DIR.glob("*.txt")):
        raw_text = file.read_text(encoding="utf-8")
        parsed = parse_teamtat_input_text(raw_text)
        text_map.update(parsed)

    return text_map


def build_user_prompt(utterance_id, text, old_annotations):
    return f"""
Utterance ID: {utterance_id}

Utterance:
{text}

Existing annotations:
{json.dumps(old_annotations, ensure_ascii=False, indent=2)}

Revise the annotations.

Output format:
{{
  "utterance_id": {utterance_id},
  "annotations": [
    {{"label": "PRODUCT", "text": "..."}},
    {{"label": "SEGMENT", "text": "..."}},
    {{"label": "GROWTH_SIGNAL", "text": "..."}}
  ]
}}
"""


def extract_json(content):
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```json", "", content)
        content = re.sub(r"^```", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()

    return json.loads(content)


def relabel_one(utterance_id, text, old_annotations):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    utterance_id,
                    text,
                    old_annotations,
                ),
            },
        ],
    )

    content = response.choices[0].message.content
    return extract_json(content)


def validate_item(item, text):
    valid_labels = {"PRODUCT", "SEGMENT", "GROWTH_SIGNAL"}
    cleaned = []
    seen = set()

    for ann in item.get("annotations", []):
        label = ann.get("label")
        span = ann.get("text", "").strip()

        if label not in valid_labels:
            continue

        if not span:
            continue

        if span not in text:
            print(f"[WARN] span not in text, removed: {span}")
            continue

        key = (label, span)
        if key in seen:
            continue

        seen.add(key)
        cleaned.append({
            "label": label,
            "text": span,
        })

    return {
        "utterance_id": item["utterance_id"],
        "annotations": cleaned,
    }


def process_file(label_file, text_map):
    data = load_json(label_file)
    results = []

    out_path = OUT_DIR / label_file.name.replace("_train.json", "_gpt.json")

    if out_path.exists():
        print(f"[SKIP] already exists: {out_path}")
        return

    for idx, item in enumerate(data, start=1):
        uid = item["utterance_id"]

        if uid not in text_map:
            raise ValueError(f"Missing text for utterance_id={uid}")

        text = text_map[uid]
        old_annotations = item.get("annotations", [])

        print(f"[{label_file.name}] {idx}/{len(data)} uid={uid}")

        revised = relabel_one(uid, text, old_annotations)
        revised = validate_item(revised, text)
        results.append(revised)

        time.sleep(SLEEP_SECONDS)

    save_json(results, out_path)
    print(f"[SAVED] {out_path}")


def main():
    text_map = load_text_map()

    # 000~009만 실행
    label_files = sorted(
        LABEL_DIR.glob("teamtat_doc_00[0-9]_train.json")
    )

    print(f"Loaded utterance texts: {len(text_map)}")
    print(f"Label files: {len(label_files)}")

    if len(label_files) != 10:
        print("[WARN] Expected 10 files: teamtat_doc_000~009_train.json")
        for f in label_files:
            print(" -", f.name)

    for label_file in label_files:
        process_file(label_file, text_map)


if __name__ == "__main__":
    main()