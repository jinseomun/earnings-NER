import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "models/qwen_earnings_semantic_span_lora"

INPUT_PATH = "data/annotation/eval/teamtat_doc_011_inputs.json"
OUTPUT_PATH = "data/annotation/eval/teamtat_doc_011_pred.json"


SYSTEM_PROMPT = (
    "Extract semantic spans from the earnings call utterance. "
    "Use only these labels: PRODUCT, SEGMENT, GROWTH_SIGNAL. "
    "Return only a valid JSON array of objects with label and text. "
    "Do not add explanations. "
    "\n\n"
    "Guideline: "
    "GROWTH_SIGNAL means a readable business phrase about growth, decline, demand, revenue, margin, outlook, adoption, recovery, momentum, capacity, inventory, or strategic progress. "
    "It should include enough context to be understood, not just one keyword. "
    "PRODUCT means a standalone product, platform, service, or technology name. "
    "SEGMENT means an official business segment or business unit. "
    "If a product is part of a larger business change phrase, label the larger phrase as GROWTH_SIGNAL. "
    "If there is no meaningful span, return []. "
    "\n\n"
    "Output format: "
    '[{"label":"GROWTH_SIGNAL","text":"revenue increased 20% sequentially"}]'

)




def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_output(text):
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return json.loads(text[start:end])
    except Exception:
        return []


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
        trust_remote_code=True
    ).to(device)

    model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)
    model.eval()

    test_data = load_json(INPUT_PATH)
    predictions = []

    for item in test_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["text"]}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)

        predictions.append({
            "utterance_id": item["utterance_id"],
            "annotations": parse_output(decoded)
        })

        print(f"Done utterance_id={item['utterance_id']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()