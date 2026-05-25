import re


MEANINGLESS_SHORT_SENTENCES = {
    "yes.", "yes",
    "no.", "no",
    "okay.", "okay",
    "thanks.", "thanks",
    "thank you.", "thank you"
}


def is_metadata_line(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    # 구분선
    if re.fullmatch(r"-{3,}", text):
        return True

    # [70]
    if re.fullmatch(r"\[\d+\]", text):
        return True

    # [70] --------
    if re.fullmatch(r"\[\d+\]\s*-{3,}", text):
        return True

    # 기호만 있는 줄
    if re.fullmatch(r"[\-\–\—_= ]+", text):
        return True

    return False


def is_meaningful_sentence(sentence: str) -> bool:
    sentence = sentence.strip()

    if is_metadata_line(sentence):
        return False

    if sentence.lower() in MEANINGLESS_SHORT_SENTENCES:
        return False

    if not re.search(r"[A-Za-z]", sentence):
        return False

    return True


def filter_sentences(sentences):
    return [
        sentence.strip()
        for sentence in sentences
        if is_meaningful_sentence(sentence)
    ]