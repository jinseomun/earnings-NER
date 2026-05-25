import re


TRAILING_METADATA_PATTERNS = [
    r"\n\s*-+\s*\n\s*Definitions\s*\n\s*-+",
    r"\n\s*-+\s*\n\s*Disclaimer\s*\n\s*-+",
    r"\n\s*Copyright\s+\d{4}\s+Thomson Reuters",
]

def remove_front_metadata(text: str) -> str:
    patterns = [
        r"Presentation\s*\n[-=]+",
        r"Operator\s*\[\d+\]"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return text[match.start():].strip()

    return text.strip()

def remove_trailing_metadata(text: str) -> str:
    for pattern in TRAILING_METADATA_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return text[:match.start()].strip()

    return text.strip()

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_transcript(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = remove_front_metadata(text)
    text = remove_trailing_metadata(text)
    text = normalize_whitespace(text)

    return text