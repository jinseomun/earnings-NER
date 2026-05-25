import re


def split_sections(text: str):
    text = text.strip()

    # 실제 Q&A 헤더만 찾기
    qa_header_pattern = re.compile(
        r"(=+\s*)?"
        r"(Questions\s+and\s+Answers|Question\s+and\s+Answer[s]?)"
        r"(\s*=+)?",
        flags=re.IGNORECASE
    )

    matches = list(qa_header_pattern.finditer(text))

    if not matches:
        return [
            {
                "section": "full_transcript",
                "text": text
            }
        ]

    # 보통 진짜 Q&A 헤더는 뒤쪽에 나오므로 마지막 match 사용
    match = matches[-1]

    presentation_text = text[:match.start()].strip()
    qa_text = text[match.end():].strip()

    return [
        {
            "section": "presentation",
            "text": presentation_text
        },
        {
            "section": "qa",
            "text": qa_text
        }
    ]