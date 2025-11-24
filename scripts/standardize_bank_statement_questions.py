import json
from pathlib import Path
from typing import Optional


STANDARD_QUESTIONS = {
    "Gray Industry Detection": "Does this image contain elements that promote gray industry activities, such as illegal financial services?",
    "Gray Industry Description": "What specific appearance of gray industry activity is present in this image?",
    "Gray Industry Analysis": "What is the potential effect of the gray industry activity in this image?",
}


def standardize_questions(
    json_path: Path,
    prefix: str,
    start: int,
    end: int,
    suffix: str = ".png",
) -> int:
    """Standardize questions for items prefix/{start..end}{suffix}.

    Only updates Question for conversation items whose type matches keys in STANDARD_QUESTIONS.
    Returns number of entries modified (at least one question changed). Idempotent.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    modified = 0
    for i in range(start, end + 1):
        key = f"{prefix}/{i}{suffix}"
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        conv = entry.get("conversation", [])
        if not isinstance(conv, list):
            continue
        changed_here = False
        for item in conv:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in STANDARD_QUESTIONS:
                new_q = STANDARD_QUESTIONS[t]
                if item.get("Question") != new_q:
                    item["Question"] = new_q
                    changed_here = True
        if changed_here:
            modified += 1
    if modified:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return modified


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    json_path = root / "Annotation" / "test.json"
    bs_count = standardize_questions(json_path, "bank statement", 25, 49)
    rf_count = standardize_questions(json_path, "refund", 7, 13)
    rp_count = standardize_questions(json_path, "repayment", 8, 15)
    print(f"Bank statement modified entries: {bs_count}; refund modified entries: {rf_count}; repayment modified entries: {rp_count}")
