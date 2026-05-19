# earnings-NER

어닝콜(Earnings Call) Transcript를 분석하여 핵심 재무 정보를 추출하는  
NER 기반 Financial Information Extraction 프로젝트입니다.

## 프로젝트 개요

기업의 어닝콜 transcript는 길이가 길고 금융 용어가 많아 일반 사용자가 이해하기 어렵다.

본 프로젝트는 어닝콜 텍스트에서 핵심 재무 정보(매출, EPS, 가이던스, 리스크 등)를 추출하고 구조화하여 사용자가 빠르게 핵심 내용을 파악할 수 있도록 하는 것을 목표로 한다.

---

## 주요 기능

- 어닝콜 transcript 전처리
- 금융 도메인 특화 NER로 핵심 재무 정보 추출
- 구조화된 JSON 출력

Financial Information Extraction

---

## Label Schema

현재 사용하는 Financial Entity Label은 다음과 같습니다.

- REVENUE
- EPS
- GUIDANCE
- RISK
- GROWTH
- DECLINE
- MARGIN
- COST
- PRODUCT
- REGION

---

## 데이터셋

Hugging Face Dataset:

https://huggingface.co/datasets/soumakchak/earnings_call_dataset

---

## 프로젝트 구조

```bash
earnings-NER/
├── data/
├── notebooks/
├── outputs/
├── src/
├── README.md
└── .gitignore
```

---

## 예시

입력 Transcript:

> "Revenue grew 14% year-over-year and we expect continued cloud growth next quarter."

출력 예시:

```json
{
  "REVENUE": "14% year-over-year growth",
  "GUIDANCE": "continued cloud growth next quarter"
}
```

---
