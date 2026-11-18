"""Score Hebrew PII detectors on shared benchmarks.

Models are scored on five shared categories (person, address,
phone, email, account) over three datasets:

  redact      Hebrew records of the REDACT benchmark (external, the one that matters)
  hard        hand-written cases in hard_set.txt
  golemguard  a sample of GolemPII's own synthetic test split, as an
              in-distribution reference

Scoring is per character and typed. Gold types that neither model is expected
to cover the same way (dates, cities, organisations, employee ids, ...) are
excluded from scoring, so predictions on them are neither rewarded nor punished.

Setup, once:
  uv sync
  accept the terms at https://huggingface.co/datasets/guneeshv/REDACT-PII-Benchmark
  uv run hf auth login

Run:
  uv run compare.py
  uv run compare.py --decoder golem --validators
  uv run compare.py --model mine=runs/mine:golempii_v2 --decoder golem
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from golempii.decode import decode_spans, is_valid

CATEGORIES = ["person", "address", "phone", "email", "account"]
IGNORE = "ignore"

# Model labels missing from a map (GolemPII's DATE and CC_PROVIDER) are dropped.
GOLEMPII_LABELS = {
    "FIRST_NAME": "person",
    "LAST_NAME": "person",
    "STREET": "address",
    "CITY": "address",
    "POSTAL_CODE": "address",
    "PHONE_NUM": "phone",
    "EMAIL": "email",
    "ID_NUM": "account",
    "BANK_ACCOUNT_NUM": "account",
    "CC_NUM": "account",
}
GOLEMPII_FINE_LABELS = [
    "FIRST_NAME", "LAST_NAME", "ID_NUM", "BANK_ACCOUNT_NUM", "CC_NUM", "CC_PROVIDER",
    "CITY", "STREET", "POSTAL_CODE", "PHONE_NUM", "EMAIL", "DATE",
]
GOLEMPII_V2_NEW_LABELS = ["PASSPORT_NUM", "DRIVER_LICENSE_NUM", "IBAN"]
GOLEMPII_V2_LABELS = {**GOLEMPII_LABELS, **{label: "account" for label in GOLEMPII_V2_NEW_LABELS}}
LABEL_MAPS = {
    "golempii": GOLEMPII_LABELS,
    "golempii_v2": GOLEMPII_V2_LABELS,
}
HAS_FINE_LABELS = {"golempii", "golempii_v2"}
DEFAULT_MODELS = [
    ("GolemPII-v1", "CordwainerSmith/GolemPII-v1", "golempii"),
]

# REDACT types missing from this map are excluded from scoring.
REDACT_LABELS = {
    "First_Given_Name": "person",
    "Last_Family_Name": "person",
    "Full_Name": "person",
    "Preferred_Name": "person",
    "Address_Personal": "address",
    "Address_Work": "address",
    "Telephone_Numbers_Personal": "phone",
    "Telephone_Numbers_Work": "phone",
    "Personal_Email_Address": "email",
    "Work_Email_Address": "email",
    "National_Identification_Number": "account",
    "Credit_Card_Numbers": "account",
    "Passport_Number": "account",
    "Driving_License_Number": "account",
    "Tax_Reference_Number": "account",
}
FULL_NAME = "FULL_NAME"
NAME_LABELS = {"FIRST_NAME", "LAST_NAME"}
REDACT_FINE_LABELS = {
    "First_Given_Name": "FIRST_NAME",
    "Last_Family_Name": "LAST_NAME",
    "National_Identification_Number": "ID_NUM",
    "Credit_Card_Numbers": "CC_NUM",
    "Passport_Number": "PASSPORT_NUM",
    "Driving_License_Number": "DRIVER_LICENSE_NUM",
    "City": "CITY",
    "Date_Time": "DATE",
    "Date_of_Birth": "DATE",
    "Telephone_Numbers_Personal": "PHONE_NUM",
    "Telephone_Numbers_Work": "PHONE_NUM",
    "Personal_Email_Address": "EMAIL",
    "Work_Email_Address": "EMAIL",
}
# Model labels with no REDACT counterpart at all, so predicting them is not judged.
REDACT_UNJUDGED_LABELS = {"STREET", "POSTAL_CODE", "CC_PROVIDER"}
ADDRESS_PART_LABELS = {"STREET", "POSTAL_CODE"}
ADDRESS_SEPARATOR = re.compile(r"[\s,]{0,3}")
CODE_SWITCHING_LEVELS = ["none", "light", "heavy"]
REDACT_REPO = "guneeshv/REDACT-PII-Benchmark"
REDACT_FILE = "pii_benchmark_full.json"
HEBREW_CODES = {"he", "hebrew", "he_il", "he-il"}

HARD_MARKER = re.compile(r"\[\[(\w+)\|(.+?)\]\]")


def record_language(record):
    # REDACT keeps the language at the top level or under "axes", depending on provenance.
    return str(record.get("language") or record["axes"]["language"])


def redact_fine_label(entity_type):
    if entity_type == "Full_Name":
        return FULL_NAME
    return REDACT_FINE_LABELS.get(entity_type, IGNORE)


def load_redact(path):
    if path is None:
        path = hf_hub_download(REDACT_REPO, REDACT_FILE, repo_type="dataset")
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    hebrew = [r for r in records if record_language(r).lower() in HEBREW_CODES]
    if not hebrew:
        languages = sorted({record_language(r) for r in records})
        raise SystemExit(f"No Hebrew records matched {HEBREW_CODES}. Languages in the file: {languages}")

    examples = []
    unaligned = 0
    for record in hebrew:
        text = record["text"]
        spans = []
        fine_spans = []
        for entity in record["entities"]:
            start, end, value = entity.get("start"), entity.get("end"), entity["entity_string"]
            if start is not None and end is not None and text[start:end] == value:
                spans.append((start, end, REDACT_LABELS.get(entity["entity_type"], IGNORE)))
                fine_spans.append((start, end, redact_fine_label(entity["entity_type"])))
                continue
            # Some REDACT entities have missing or misaligned offsets. Exclude every
            # occurrence of their text from scoring instead of trusting the offsets.
            unaligned += 1
            for match in re.finditer(re.escape(value), text) if value else []:
                spans.append((match.start(), match.end(), IGNORE))
                fine_spans.append((match.start(), match.end(), IGNORE))
        examples.append({
            "text": text,
            "spans": spans,
            "fine_spans": fine_spans,
            "code_switching": record["axes"]["code_switching"],
        })
    total = sum(len(record["entities"]) for record in hebrew)
    print(f"REDACT Hebrew: {len(examples)} records, {unaligned} of {total} entities unaligned and excluded")
    return examples


def split_redact(examples, part, seed=0):
    if part == "all":
        return examples
    by_level = defaultdict(list)
    for index, example in enumerate(examples):
        by_level[example["code_switching"]].append(index)
    rng = random.Random(seed)
    chosen = []
    for level in sorted(by_level):
        indices = by_level[level]
        rng.shuffle(indices)
        half = len(indices) // 2
        chosen.extend(indices[:half] if part == "select" else indices[half:])
    return [examples[index] for index in sorted(chosen)]


def load_hard_set(path):
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        text = ""
        spans = []
        position = 0
        for match in HARD_MARKER.finditer(line):
            category, value = match.group(1), match.group(2)
            if category not in CATEGORIES:
                raise SystemExit(f"Unknown category '{category}' in {path}: {line}")
            text += line[position:match.start()]
            spans.append((len(text), len(text) + len(value), category))
            text += value
            position = match.end()
        text += line[position:]
        examples.append({"text": text, "spans": spans})
    return examples


def load_golemguard(sample_size):
    dataset = load_dataset("CordwainerSmith/GolemGuard", split="test", streaming=True)
    dataset = dataset.select_columns(["source_text", "privacy_mask"])
    dataset = dataset.shuffle(seed=0, buffer_size=2000).take(sample_size)
    examples = []
    for row in dataset:
        spans = [
            (item["start"], item["end"], GOLEMPII_LABELS.get(item["label"], IGNORE))
            for item in row["privacy_mask"]
        ]
        fine_spans = [(item["start"], item["end"], item["label"]) for item in row["privacy_mask"]]
        examples.append({"text": row["source_text"], "spans": spans, "fine_spans": fine_spans})
    return examples


def load_pipeline_predictor(model_id, batch_size, device, validators):
    # stride lets the pipeline chunk texts longer than the 512-token window.
    pipe = pipeline(
        "token-classification",
        model=model_id,
        aggregation_strategy="simple",
        stride=128,
        device=device,
    )

    def predict(texts):
        outputs = pipe(texts, batch_size=batch_size)
        predictions = []
        for text, output in zip(texts, outputs):
            spans = [(item["start"], item["end"], item["entity_group"]) for item in output]
            if validators:
                spans = [span for span in spans if is_valid(span[2], text[span[0]:span[1]])]
            predictions.append(spans)
        return predictions

    return predict


def load_golem_predictor(model_id, batch_size, device, validators):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
    return lambda texts: predict_golem(tokenizer, model, texts, batch_size, device, validators)


def predict_golem(tokenizer, model, texts, batch_size, device, validators):
    id2label = model.config.id2label
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    window_texts = encoded["overflow_to_sample_mapping"]
    order = sorted(range(len(window_texts)), key=lambda window: len(encoded["input_ids"][window]))

    # Per text, the best label of each token: the window where the token is farthest from the window edge.
    best = [{} for _ in texts]
    for first in range(0, len(order), batch_size):
        batch = order[first:first + batch_size]
        padded = tokenizer.pad(
            {
                "input_ids": [encoded["input_ids"][window] for window in batch],
                "attention_mask": [encoded["attention_mask"][window] for window in batch],
            },
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            label_ids = model(**padded).logits.argmax(dim=-1).tolist()
        for window, window_label_ids in zip(batch, label_ids):
            offsets = encoded["offset_mapping"][window]
            real = [index for index, offset in enumerate(offsets) if offset[0] != offset[1]]
            for rank, index in enumerate(real):
                distance = min(rank, len(real) - 1 - rank)
                key = tuple(offsets[index])
                if key not in best[window_texts[window]] or distance > best[window_texts[window]][key][0]:
                    best[window_texts[window]][key] = (distance, id2label[window_label_ids[index]])

    predictions = []
    for text, token_labels in zip(texts, best):
        offsets = sorted(token_labels)
        spans = decode_spans(text, offsets, [token_labels[offset][1] for offset in offsets])
        if validators:
            spans = [span for span in spans if is_valid(span[2], text[span[0]:span[1]])]
        predictions.append(spans)
    return predictions


def coarse_spans(predictions, label_map):
    return [
        [(start, end, label_map[label]) for start, end, label in spans if label in label_map]
        for spans in predictions
    ]


def drop_lone_cities(spans, text):
    # REDACT counts a city as an address only inside one; a city on its own is its unscored City type.
    ordered = sorted(spans)
    kept = []
    for index, (start, end, label) in enumerate(ordered):
        if label == "CITY":
            before = ordered[index - 1] if index > 0 else None
            after = ordered[index + 1] if index + 1 < len(ordered) else None
            joined = (
                before is not None
                and before[2] in ADDRESS_PART_LABELS
                and ADDRESS_SEPARATOR.fullmatch(text[before[1]:start]) is not None
            ) or (
                after is not None
                and after[2] in ADDRESS_PART_LABELS
                and ADDRESS_SEPARATOR.fullmatch(text[end:after[0]]) is not None
            )
            if not joined:
                continue
        kept.append((start, end, label))
    return kept


def fine_predictions(predictions, dataset_name):
    unjudged = REDACT_UNJUDGED_LABELS if dataset_name == "redact" else set()
    return [[span for span in spans if span[2] not in unjudged] for spans in predictions]


def char_labels(length, spans):
    labels = [None] * length
    # Scored categories are written last so they win over an enclosing ignored span.
    for start, end, category in sorted(spans, key=lambda span: span[2] != IGNORE):
        for index in range(start, min(end, length)):
            labels[index] = category
    return labels


def tally(examples, predictions, span_key, label_matches):
    counts = defaultdict(lambda: defaultdict(int))
    for example, predicted_spans in zip(examples, predictions):
        text = example["text"]
        gold = char_labels(len(text), example[span_key])
        predicted = char_labels(len(text), predicted_spans)

        for index, char in enumerate(text):
            if char.isspace() or gold[index] == IGNORE:
                continue
            gold_category, predicted_category = gold[index], predicted[index]
            if predicted_category is not None and label_matches(gold_category, predicted_category):
                counts[predicted_category]["tp"] += 1
                continue
            if predicted_category is not None:
                counts[predicted_category]["fp"] += 1
            if gold_category is not None:
                counts[gold_category]["fn"] += 1

        for start, end, category in example[span_key]:
            if category == IGNORE:
                continue
            counts[category]["entities"] += 1
            if any(
                predicted[index] is not None and label_matches(category, predicted[index])
                for index in range(start, min(end, len(text)))
            ):
                counts[category]["caught"] += 1
    return counts


def summarize_counts(counts, categories):
    results = {category: summarize(counts[category]) for category in categories}
    total = defaultdict(int)
    for category in categories:
        for key, value in counts[category].items():
            total[key] += value
    results["micro"] = summarize(total)
    return results


def score(examples, predictions):
    counts = tally(examples, predictions, "spans", lambda gold, predicted: predicted == gold)
    return summarize_counts(counts, CATEGORIES)


def fine_label_matches(gold, predicted):
    if gold == FULL_NAME:
        return predicted in NAME_LABELS
    return predicted == gold


def score_fine(examples, predictions):
    counts = tally(examples, predictions, "fine_spans", fine_label_matches)
    return summarize_counts(counts, sorted(counts))


def summarize(count):
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "entities": count["entities"],
        "entities_caught": count["caught"] / count["entities"] if count["entities"] else 0.0,
    }


def print_report(results, sizes):
    for dataset_name, per_model in results.items():
        print(f"\n## {dataset_name} ({sizes[dataset_name]} texts)\n")
        print("| Model | Category | Entities | Entities caught | Char precision | Char recall | Char F1 |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for model_name, scores in per_model.items():
            for category in ["micro"] + CATEGORIES:
                row = scores[category]
                print(
                    f"| {model_name} | {category} | {row['entities']} | {row['entities_caught']:.3f} "
                    f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
                )
        seconds = ", ".join(f"{name} {scores['seconds']:.0f}s" for name, scores in per_model.items())
        print(f"\nInference time: {seconds}")

        fine_models = {name: scores["fine"] for name, scores in per_model.items() if "fine" in scores}
        if fine_models:
            print(f"\n### {dataset_name}, fine labels\n")
            print("| Model | Label | Entities | Entities caught | Char precision | Char recall | Char F1 |")
            print("|---|---|---:|---:|---:|---:|---:|")
            for model_name, fine in fine_models.items():
                labels = ["micro"] + [label for label in fine if label != "micro"]
                for label in labels:
                    row = fine[label]
                    print(
                        f"| {model_name} | {label} | {row['entities']} | {row['entities_caught']:.3f} "
                        f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
                    )

        split_models = {name: scores["code_switching"] for name, scores in per_model.items() if "code_switching" in scores}
        if split_models:
            print(f"\n### {dataset_name}, by code-switching level (coarse micro)\n")
            print("| Model | Code-switching | Texts | Char precision | Char recall | Char F1 |")
            print("|---|---|---:|---:|---:|---:|")
            for model_name, groups in split_models.items():
                for level, group in groups.items():
                    print(
                        f"| {model_name} | {level} | {group['texts']} | {group['precision']:.3f} "
                        f"| {group['recall']:.3f} | {group['f1']:.3f} |"
                    )


def parse_model_arg(value):
    name, separator, location = value.partition("=")
    path, _, label_map = location.rpartition(":")
    if not separator or not name or not path or label_map not in LABEL_MAPS:
        raise argparse.ArgumentTypeError(
            f"expected NAME=PATH_OR_HUB_ID:LABEL_MAP with LABEL_MAP one of {sorted(LABEL_MAPS)}, got '{value}'"
        )
    return name, path, label_map


def score_by_code_switching(examples, coarse):
    groups = {}
    for level in CODE_SWITCHING_LEVELS:
        indices = [index for index, example in enumerate(examples) if example["code_switching"] == level]
        group = score([examples[index] for index in indices], [coarse[index] for index in indices])["micro"]
        groups[level] = {**group, "texts": len(indices)}
    return groups


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redact", help="path to a local REDACT json file (default: download from the Hub)")
    parser.add_argument("--skip-redact", action="store_true", help="run without the gated REDACT benchmark")
    parser.add_argument("--hard", default=here / "hard_set.txt")
    parser.add_argument("--golemguard-sample", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", default=here / "results.json")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=parse_model_arg,
        help=f"extra model as NAME=PATH_OR_HUB_ID:LABEL_MAP, repeatable. LABEL_MAP: {', '.join(LABEL_MAPS)}",
    )
    parser.add_argument(
        "--decoder",
        choices=["pipeline", "golem"],
        default="pipeline",
        help="pipeline: transformers aggregation. golem: direct model run with word-level decoding",
    )
    parser.add_argument("--validators", action="store_true", help="drop spans failing structural checks")
    parser.add_argument(
        "--redact-split",
        choices=["all", "select", "final"],
        default="all",
        help="select: half used for model decisions. final: held-out half, scored once for the shipped model",
    )
    parser.add_argument("--skip-defaults", action="store_true", help="score only the models passed with --model")
    args = parser.parse_args()

    datasets = {}
    if not args.skip_redact:
        datasets["redact"] = split_redact(load_redact(args.redact), args.redact_split)
    datasets["hard"] = load_hard_set(args.hard)
    datasets["golemguard"] = load_golemguard(args.golemguard_sample)
    sizes = {name: len(examples) for name, examples in datasets.items()}
    print(
        f"Datasets: {sizes}. REDACT split: {args.redact_split}. Device: {args.device}. "
        f"Decoder: {args.decoder}. Validators: {args.validators}"
    )

    results = {name: {} for name in datasets}
    models = args.model if args.skip_defaults else DEFAULT_MODELS + args.model
    if not models:
        parser.error("--skip-defaults needs at least one --model")
    for model_name, model_id, label_map_name in models:
        label_map = LABEL_MAPS[label_map_name]
        print(f"Loading {model_id}")
        if args.decoder == "golem":
            predict = load_golem_predictor(model_id, args.batch_size, args.device, args.validators)
        else:
            predict = load_pipeline_predictor(model_id, args.batch_size, args.device, args.validators)
        for dataset_name, examples in datasets.items():
            started = time.perf_counter()
            predictions = predict([e["text"] for e in examples])
            if dataset_name == "redact" and "CITY" in label_map:
                address_predictions = [drop_lone_cities(spans, e["text"]) for spans, e in zip(predictions, examples)]
                coarse = coarse_spans(address_predictions, label_map)
            else:
                coarse = coarse_spans(predictions, label_map)
            scores = score(examples, coarse)
            scores["seconds"] = time.perf_counter() - started
            if label_map_name in HAS_FINE_LABELS and "fine_spans" in examples[0]:
                scores["fine"] = score_fine(examples, fine_predictions(predictions, dataset_name))
            if dataset_name == "redact":
                scores["code_switching"] = score_by_code_switching(examples, coarse)
            results[dataset_name][model_name] = scores
        del predict

    print_report(results, sizes)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
