"""Two checks that do not depend on REDACT's span-boundary conventions.

1. REDACT `select`, entity level: a gold entity counts as found when a predicted span of its category
   touches it, and a predicted span counts as right when it touches a gold entity of its category.
   Where a span starts and ends plays no role, so following or ignoring REDACT's boundary habits
   (street word, titles, "ending" phrases) does not move these numbers.
2. GolemGuard test rows (v1's own dataset, never used to tune v2): per-label entity recall and span
   precision, names split by whether our lexicons contain them, and the predicted card numbers that
   GolemGuard does not label checked for a valid card checksum.

Run:
  uv run independent_checks.py --model v1=CordwainerSmith/GolemPII-v1:golempii --model v2=CordwainerSmith/GolemPII-v2-base:golempii_v2

Predictions are cached in logs/preds/, so adding a model only runs that model.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from compare import (
    CATEGORIES,
    GOLEMPII_LABELS,
    HAS_FINE_LABELS,
    IGNORE,
    LABEL_MAPS,
    coarse_spans,
    default_device,
    drop_lone_cities,
    load_pipeline_predictor,
    load_redact,
    parse_model_arg,
    split_redact,
)
from data_gen import fillers
from golempii.decode import digits_of, luhn_ok

CACHE_DIR = Path(__file__).parent / "logs" / "preds"
NAME_LABELS = ["FIRST_NAME", "LAST_NAME"]
LEXICON_FILES = {"FIRST_NAME": "first_names_he.txt", "LAST_NAME": "last_names_he.txt"}
NAME_GROUPS = ["old lexicon", "grown lexicon only", "held out", "in no lexicon"]


def load_golemguard_rows(sample_size):
    dataset = load_dataset("CordwainerSmith/GolemGuard", split="test", streaming=True)
    dataset = dataset.select_columns(["source_text", "privacy_mask"])
    dataset = dataset.shuffle(seed=0, buffer_size=2000).take(sample_size)
    return [
        {"text": row["source_text"], "gold": [(item["start"], item["end"], item["label"]) for item in row["privacy_mask"]]}
        for row in dataset
    ]


def cached_predictions(dataset_name, model_name, model_id, texts, device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{dataset_name}_{model_name}.pkl"
    if path.exists():
        cached = pickle.loads(path.read_bytes())
        if cached["model_id"] == model_id and len(cached["predictions"]) == len(texts):
            return cached["predictions"]
    predictions = load_pipeline_predictor(model_id, 8, device, False)(texts)
    path.write_bytes(pickle.dumps({"model_id": model_id, "predictions": predictions}))
    return predictions


def overlaps(start, end, spans):
    return [span for span in spans if span[0] < end and start < span[1]]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def entity_scores(gold_by_example, predicted_by_example, labels):
    """Gold spans labelled IGNORE are not entities, and a predicted span that touches only them is not judged."""
    counts = {label: defaultdict(int) for label in labels}
    for gold, predicted in zip(gold_by_example, predicted_by_example):
        for start, end, label in gold:
            if label not in counts:
                continue
            counts[label]["entities"] += 1
            if any(span[2] == label for span in overlaps(start, end, predicted)):
                counts[label]["found"] += 1
        for start, end, label in predicted:
            if label not in counts:
                continue
            touched = overlaps(start, end, gold)
            if touched and all(span[2] == IGNORE for span in touched):
                continue
            counts[label]["spans"] += 1
            if any(span[2] == label for span in touched):
                counts[label]["right"] += 1
    total = defaultdict(int)
    for label in labels:
        for key, value in counts[label].items():
            total[key] += value
    counts["micro"] = total
    results = {}
    for label, count in counts.items():
        precision, recall = ratio(count["right"], count["spans"]), ratio(count["found"], count["entities"])
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "entities": count["entities"],
            "spans": count["spans"],
        }
    return results


def redact_check(examples, predictions, label_map):
    if "CITY" in label_map:
        predictions = [drop_lone_cities(spans, example["text"]) for spans, example in zip(predictions, examples)]
    coarse = coarse_spans(predictions, label_map)
    return entity_scores([example["spans"] for example in examples], coarse, CATEGORIES)


def name_group(label, value, lexicons):
    value = value.strip()
    if value in lexicons[label]["old"]:
        return "old lexicon"
    if value in lexicons[label]["train"]:
        return "grown lexicon only"
    if value in lexicons[label]["heldout"]:
        return "held out"
    return "in no lexicon"


def load_name_lexicons():
    lexicons = {}
    for label, file_name in LEXICON_FILES.items():
        lexicons[label] = {
            "old": {hebrew for hebrew, _ in fillers.load_lexicon(f"old/{file_name}")},
            "train": {hebrew for hebrew, _ in fillers.load_lexicon(file_name)},
            "heldout": {hebrew for hebrew, _ in fillers.load_lexicon(f"heldout_{file_name}")},
        }
    return lexicons


def golemguard_check(rows, predictions, label_map_name, lexicons):
    label_map = LABEL_MAPS[label_map_name]
    coarse_gold = [[(start, end, GOLEMPII_LABELS.get(label, IGNORE)) for start, end, label in row["gold"]] for row in rows]
    result = {"coarse": entity_scores(coarse_gold, coarse_spans(predictions, label_map), CATEGORIES)}
    if label_map_name not in HAS_FINE_LABELS:
        return result

    gold_labels = sorted({label for row in rows for _, _, label in row["gold"]})
    result["fine"] = entity_scores([row["gold"] for row in rows], predictions, gold_labels)

    # Names: found means touched by a FIRST_NAME or LAST_NAME span; typed means touched by the right one of the two.
    names = {label: {group: defaultdict(int) for group in NAME_GROUPS} for label in NAME_LABELS}
    unlabelled_cards = defaultdict(int)
    v2_only = defaultdict(int)
    for row, predicted in zip(rows, predictions):
        text = row["text"]
        for start, end, label in row["gold"]:
            if label not in NAME_LABELS:
                continue
            count = names[label][name_group(label, text[start:end], lexicons)]
            touched = overlaps(start, end, predicted)
            count["entities"] += 1
            count["found"] += any(span[2] in NAME_LABELS for span in touched)
            count["typed"] += any(span[2] == label for span in touched)
        for start, end, label in predicted:
            if overlaps(start, end, row["gold"]):
                continue
            if label == "CC_NUM":
                unlabelled_cards["spans"] += 1
                digits = digits_of(text[start:end])
                unlabelled_cards["valid_checksum"] += (len(digits) >= 14 and luhn_ok(digits)) or (
                    len(digits) in (8, 9) and fillers.is_isracard_valid(digits)
                )
            elif label not in gold_labels:
                v2_only[label] += 1
    result["names"] = {
        label: {
            group: {
                "entities": count["entities"],
                "found": ratio(count["found"], count["entities"]),
                "typed": ratio(count["typed"], count["entities"]),
            }
            for group, count in groups.items()
        }
        for label, groups in names.items()
    }
    result["cc_num_spans_on_unlabelled_text"] = dict(unlabelled_cards)
    result["spans_of_labels_golemguard_does_not_have"] = dict(v2_only)
    return result


def print_tables(results, models, redact_split="select"):
    names = [name for name, _, _ in models]
    print(f"\n## REDACT `{redact_split}`, entity level (precision / recall / F1)\n")
    print("| Model | " + " | ".join(CATEGORIES + ["micro"]) + " |")
    print("|---|" + "---|" * (len(CATEGORIES) + 1))
    for name in names:
        scores = results[f"redact_{redact_split}_entity"][name]
        cells = [f"{scores[c]['precision']:.3f} / {scores[c]['recall']:.3f} / {scores[c]['f1']:.3f}" for c in CATEGORIES + ["micro"]]
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## GolemGuard test rows, entity level, coarse (precision / recall / F1)\n")
    print("| Model | " + " | ".join(CATEGORIES + ["micro"]) + " |")
    print("|---|" + "---|" * (len(CATEGORIES) + 1))
    for name in names:
        scores = results["golemguard"][name]["coarse"]
        cells = [f"{scores[c]['precision']:.3f} / {scores[c]['recall']:.3f} / {scores[c]['f1']:.3f}" for c in CATEGORIES + ["micro"]]
        print(f"| {name} | " + " | ".join(cells) + " |")

    fine_names = [name for name in names if "fine" in results["golemguard"][name]]
    if not fine_names:
        return
    labels = [label for label in results["golemguard"][fine_names[0]]["fine"] if label != "micro"] + ["micro"]
    print("\n## GolemGuard test rows, per label (span precision / entity recall)\n")
    print("| Label | Entities | " + " | ".join(fine_names) + " |")
    print("|---|---:|" + "---|" * len(fine_names))
    for label in labels:
        cells = [
            f"{results['golemguard'][name]['fine'][label]['precision']:.3f} / {results['golemguard'][name]['fine'][label]['recall']:.3f}"
            for name in fine_names
        ]
        print(f"| {label} | {results['golemguard'][fine_names[0]]['fine'][label]['entities']} | " + " | ".join(cells) + " |")

    print("\n## GolemGuard names by lexicon membership (found as a name / given the right one of FIRST and LAST)\n")
    print("| Label | Group | Entities | " + " | ".join(fine_names) + " |")
    print("|---|---|---:|" + "---|" * len(fine_names))
    for label in NAME_LABELS:
        for group in NAME_GROUPS:
            rows = [results["golemguard"][name]["names"][label][group] for name in fine_names]
            cells = [f"{row['found']:.3f} / {row['typed']:.3f}" for row in rows]
            print(f"| {label} | {group} | {rows[0]['entities']} | " + " | ".join(cells) + " |")

    print("\n## GolemGuard: CC_NUM spans on text GolemGuard does not label\n")
    for name in fine_names:
        cards = results["golemguard"][name]["cc_num_spans_on_unlabelled_text"]
        print(f"- {name}: {cards.get('spans', 0)} spans, {cards.get('valid_checksum', 0)} with a valid card checksum")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg, help="NAME=PATH_OR_HUB_ID:LABEL_MAP, repeatable")
    parser.add_argument("--golemguard-sample", type=int, default=3000)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--redact-split", choices=["select", "final"], default="select", help="final: held-out half, scored once for the shipped model")
    parser.add_argument("--out", default=Path(__file__).parent / "results_independent.json")
    args = parser.parse_args()

    redact = split_redact(load_redact(None), args.redact_split)
    redact_key = f"redact_{args.redact_split}_entity"
    golemguard = load_golemguard_rows(args.golemguard_sample)
    lexicons = load_name_lexicons()
    results = {redact_key: {}, "golemguard": {}, "golemguard_rows": len(golemguard)}
    for model_name, model_id, label_map_name in args.model:
        redact_predictions = cached_predictions(f"redact_{args.redact_split}", model_name, model_id, [e["text"] for e in redact], args.device)
        results[redact_key][model_name] = redact_check(redact, redact_predictions, LABEL_MAPS[label_map_name])
        golemguard_predictions = cached_predictions(
            f"golemguard{args.golemguard_sample}", model_name, model_id, [row["text"] for row in golemguard], args.device
        )
        results["golemguard"][model_name] = golemguard_check(golemguard, golemguard_predictions, label_map_name, lexicons)
    print_tables(results, args.model, args.redact_split)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
