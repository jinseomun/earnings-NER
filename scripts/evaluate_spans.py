import json
import re
from collections import Counter


GOLD_PATH = "data/annotation/gold/teamtat_doc_011_gold.json"
PRED_PATH = "data/annotation/eval/teamtat_doc_011_pred.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", normalize_text(text))


def span_key(annotation):
    return (
        annotation.get("label", ""),
        normalize_text(annotation.get("text", ""))
    )


def compute_prf(tp, pred_total, gold_total):
    precision = tp / pred_total if pred_total else 0
    recall = tp / gold_total if gold_total else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    return precision, recall, f1


def to_map(data):
    return {
        item["utterance_id"]: item.get("annotations", [])
        for item in data
    }


def entity_level_eval(gold_map, pred_map, all_ids):
    tp = pred_total = gold_total = 0
    labels = set()

    for uid in all_ids:
        gold_set = set(span_key(a) for a in gold_map.get(uid, []))
        pred_set = set(span_key(a) for a in pred_map.get(uid, []))

        tp += len(gold_set & pred_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)

        labels.update(label for label, _ in gold_set)
        labels.update(label for label, _ in pred_set)

    return compute_prf(tp, pred_total, gold_total)


def token_level_eval(gold_map, pred_map, all_ids):
    labels = set()
    for uid in all_ids:
        labels.update(a.get("label", "") for a in gold_map.get(uid, []))
        labels.update(a.get("label", "") for a in pred_map.get(uid, []))

    total_tp = total_pred = total_gold = 0

    for label in labels:
        for uid in all_ids:
            gold_tokens = Counter()
            pred_tokens = Counter()

            for ann in gold_map.get(uid, []):
                if ann.get("label") == label:
                    gold_tokens.update(tokenize(ann.get("text", "")))

            for ann in pred_map.get(uid, []):
                if ann.get("label") == label:
                    pred_tokens.update(tokenize(ann.get("text", "")))

            overlap = gold_tokens & pred_tokens

            total_tp += sum(overlap.values())
            total_gold += sum(gold_tokens.values())
            total_pred += sum(pred_tokens.values())

    return compute_prf(total_tp, total_pred, total_gold)


def jaccard(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)

    if not set1 and not set2:
        return 1.0

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / len(set1 | set2)


def partial_entity_eval(gold_map, pred_map, all_ids, threshold=0.5):
    """
    같은 label끼리 token Jaccard가 threshold 이상이면 partial match로 인정.
    한 pred는 한 gold에만 매칭되도록 greedy matching.
    """
    tp = pred_total = gold_total = 0
    jaccard_scores = []

    for uid in all_ids:
        gold = gold_map.get(uid, [])
        pred = pred_map.get(uid, [])

        gold_total += len(gold)
        pred_total += len(pred)

        used_gold = set()

        for p in pred:
            best_idx = None
            best_score = 0.0

            for gi, g in enumerate(gold):
                if gi in used_gold:
                    continue

                if p.get("label") != g.get("label"):
                    continue

                score = jaccard(
                    tokenize(p.get("text", "")),
                    tokenize(g.get("text", ""))
                )

                if score > best_score:
                    best_score = score
                    best_idx = gi

            if best_idx is not None and best_score >= threshold:
                tp += 1
                used_gold.add(best_idx)
                jaccard_scores.append(best_score)

    precision, recall, f1 = compute_prf(tp, pred_total, gold_total)
    avg_jaccard = (
        sum(jaccard_scores) / len(jaccard_scores)
        if jaccard_scores
        else 0
    )

    return precision, recall, f1, avg_jaccard


def average_best_jaccard(gold_map, pred_map, all_ids):
    """
    gold span마다 같은 label의 pred span 중 가장 높은 Jaccard를 구해서 평균.
    """
    scores = []

    for uid in all_ids:
        gold = gold_map.get(uid, [])
        pred = pred_map.get(uid, [])

        for g in gold:
            best = 0.0

            for p in pred:
                if p.get("label") != g.get("label"):
                    continue

                score = jaccard(
                    tokenize(g.get("text", "")),
                    tokenize(p.get("text", ""))
                )

                best = max(best, score)

            scores.append(best)

    return sum(scores) / len(scores) if scores else 0


def over_generation_stats(gold_map, pred_map, all_ids):
    gold_count = sum(len(gold_map.get(uid, [])) for uid in all_ids)
    pred_count = sum(len(pred_map.get(uid, [])) for uid in all_ids)

    ratio = pred_count / gold_count if gold_count else 0

    return gold_count, pred_count, ratio


def no_tag_accuracy(gold_map, pred_map, all_ids):
    no_tag_total = 0
    no_tag_correct = 0

    for uid in all_ids:
        gold_empty = len(gold_map.get(uid, [])) == 0
        pred_empty = len(pred_map.get(uid, [])) == 0

        if gold_empty:
            no_tag_total += 1
            if pred_empty:
                no_tag_correct += 1

    acc = no_tag_correct / no_tag_total if no_tag_total else 0

    return no_tag_correct, no_tag_total, acc


def invalid_annotation_rate(pred_map, all_ids):
    valid_labels = {"PRODUCT", "SEGMENT", "GROWTH_SIGNAL"}

    total = 0
    invalid = 0

    for uid in all_ids:
        for ann in pred_map.get(uid, []):
            total += 1

            if not isinstance(ann, dict):
                invalid += 1
                continue

            if ann.get("label") not in valid_labels:
                invalid += 1
                continue

            if not isinstance(ann.get("text"), str):
                invalid += 1
                continue

            if not ann.get("text", "").strip():
                invalid += 1
                continue

    rate = invalid / total if total else 0

    return invalid, total, rate


def print_prf(title, scores):
    p, r, f1 = scores
    print(f"\n=== {title} ===")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")


def main():
    gold_data = load_json(GOLD_PATH)
    pred_data = load_json(PRED_PATH)

    gold_map = to_map(gold_data)
    pred_map = to_map(pred_data)

    all_ids = sorted(set(gold_map.keys()) | set(pred_map.keys()))

    entity_scores = entity_level_eval(gold_map, pred_map, all_ids)
    token_scores = token_level_eval(gold_map, pred_map, all_ids)
    partial_scores = partial_entity_eval(
        gold_map,
        pred_map,
        all_ids,
        threshold=0.5
    )

    avg_best_jaccard = average_best_jaccard(gold_map, pred_map, all_ids)
    gold_count, pred_count, overgen_ratio = over_generation_stats(
        gold_map,
        pred_map,
        all_ids
    )

    no_tag_correct, no_tag_total, no_tag_acc = no_tag_accuracy(
        gold_map,
        pred_map,
        all_ids
    )

    invalid_count, pred_ann_total, invalid_rate = invalid_annotation_rate(
        pred_map,
        all_ids
    )

    print_prf("Entity-level Exact Match", entity_scores)
    print_prf("Token-level Partial Match", token_scores)

    print("\n=== Partial Entity Match @ Jaccard >= 0.5 ===")
    print(f"Precision:   {partial_scores[0]:.4f}")
    print(f"Recall:      {partial_scores[1]:.4f}")
    print(f"F1:          {partial_scores[2]:.4f}")
    print(f"Avg Jaccard: {partial_scores[3]:.4f}")

    print("\n=== Average Best Jaccard ===")
    print(f"Avg Best Jaccard: {avg_best_jaccard:.4f}")

    print("\n=== Over-generation Stats ===")
    print(f"Gold spans: {gold_count}")
    print(f"Pred spans: {pred_count}")
    print(f"Pred/Gold Ratio: {overgen_ratio:.4f}")

    print("\n=== No-tag Accuracy ===")
    print(f"No-tag correct: {no_tag_correct}/{no_tag_total}")
    print(f"No-tag Accuracy: {no_tag_acc:.4f}")

    print("\n=== Invalid Annotation Rate ===")
    print(f"Invalid annotations: {invalid_count}/{pred_ann_total}")
    print(f"Invalid Rate: {invalid_rate:.4f}")


if __name__ == "__main__":
    main()