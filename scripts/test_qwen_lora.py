import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "models/qwen_earnings_semantic_span_lora"


SYSTEM_PROMPT = (
    "Extract readable business semantic spans from the earnings call utterance. "
    "Use only these labels: PRODUCT, SEGMENT, GROWTH_SIGNAL. "
    "Return a JSON array of objects with label and text. "
    "Do not add explanations."
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    utterance = (
        "Data Center revenue was $4 billion, up 5% year over year. "
        "The Data Center Group had operating profit of $1.8 billion, down 4% year over year."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": utterance}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()