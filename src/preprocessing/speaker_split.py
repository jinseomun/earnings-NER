import re


def clean_speaker_name(header: str) -> str:
    header = header.strip()

    # [2] 같은 발화 번호 제거
    header = re.sub(r"\[\d+\]", "", header).strip()

    # "Michael Spencer, Microsoft Corporation - General Manager of IR"
    # -> "Michael Spencer"
    if "," in header:
        return header.split(",")[0].strip()

    # "Operator [1]" -> "Operator"
    return header.strip()


def split_speakers(section_items):
    results = []

    # 패턴:
    # --------------------------------------------------------------------------------
    # Speaker Name, Company - Role [2]
    # --------------------------------------------------------------------------------
    speaker_pattern = re.compile(
        r"(?:^|\n)-{5,}\n"
        r"(?P<speaker>[^\n]+?)"
        r"(?:\s*\[\d+\])?"
        r"\n-{5,}",
        flags=re.MULTILINE
    )

    for item in section_items:
        doc_id = item.get("doc_id")
        section = item["section"]
        text = item["text"]

        matches = list(speaker_pattern.finditer(text))

        if not matches:
            results.append({
                "doc_id": doc_id,
                "section": section,
                "speaker": "unknown",
                "text": text.strip()
            })
            continue

        for i, match in enumerate(matches):
            speaker_header = match.group("speaker").strip()
            speaker = clean_speaker_name(speaker_header)

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            utterance = text[start:end].strip()

            if utterance:
                results.append({
                    "doc_id": doc_id,
                    "section": section,
                    "speaker": speaker,
                    "speaker_header": speaker_header,
                    "text": utterance
                })

    return results