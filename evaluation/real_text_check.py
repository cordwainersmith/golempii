"""Person names and false alarms on real Hebrew text that nobody on this project wrote or trained on.

Two human-labelled NER corpora, used for testing only:
- NEMO (Haaretz sentences, all three splits; files from imvladikon/nemo_corpus, spmrl token-single).
- IAHLT (news, Wikipedia, Knesset): the documents whose doc_id hash falls in the test fifth.
  Those documents must never be used for training (see iahlt_is_test_document).

Scored at entity level, so span boundaries play no role:
- person: a gold PER entity is found when a person span touches it; a person span is right when
  it touches a gold PER entity.
- false alarms: phone, email and account spans. News and encyclopedia text has almost no such
  PII, so nearly every one is a mistake; samples are written to the output file for reading.
- what the wrong person spans and the address spans land on (organisation, place, title, nothing).

Run:
  uv run real_text_check.py --model v1=CordwainerSmith/GolemPII-v1:golempii --model v2=CordwainerSmith/GolemPII-v2-base:golempii_v2
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from compare import LABEL_MAPS, coarse_spans, default_device, parse_model_arg
from independent_checks import cached_predictions, f1, overlaps, ratio

DATA_DIR = Path(__file__).parent / "data" / "real_text"
FALSE_ALARM_CATEGORIES = ["phone", "email", "account"]
SAMPLES_PER_CATEGORY = 40


def load_nemo():
    examples = []
    for split in ["train", "dev", "test"]:
        tokens, tags = [], []
        lines = (DATA_DIR / f"nemo_token-single_{split}.bmes").read_text(encoding="utf-8").split("\n")
        for line in lines + [""]:
            if line.strip():
                token, tag = line.rsplit(" ", 1)
                tokens.append(token)
                tags.append(tag)
            elif tokens:
                examples.append(nemo_example(tokens, tags))
                tokens, tags = [], []
    return examples


def nemo_example(tokens, tags):
    text = " ".join(tokens)
    gold = []
    position = 0
    entity_start = None
    for token, tag in zip(tokens, tags):
        end = position + len(token)
        prefix, _, label = tag.partition("-")
        if prefix in ("B", "S"):
            entity_start = position
        if prefix in ("E", "S"):
            gold.append((entity_start, end, label))
        position = end + 1
    return {"text": text, "gold": gold}


def iahlt_is_test_document(doc_id):
    return int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 5 == 0


def load_iahlt_test():
    examples = []
    with (DATA_DIR / "iahlt_unique.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            if iahlt_is_test_document(record["metadata"]["doc_id"]):
                gold = [(start, end, label) for start, end, label in record["label"]]
                examples.append({"text": record["text"], "gold": gold})
    return examples


def merge_adjacent_person_spans(spans, text):
    """A first and a last name side by side are one person, so models with one name label and with two are counted alike."""
    merged = []
    for start, end, category in sorted(spans):
        previous = merged[-1] if merged else None
        if previous and category == "person" and previous[2] == "person" and not text[previous[1] : start].strip():
            merged[-1] = (previous[0], end, "person")
        else:
            merged.append((start, end, category))
    return merged


def check(examples, predictions, label_map):
    coarse = [merge_adjacent_person_spans(spans, e["text"]) for spans, e in zip(coarse_spans(predictions, label_map), examples)]
    entities = found = person_spans = right = 0
    wrong_person_on = Counter()
    address_on = Counter()
    false_alarms = Counter()
    samples = {category: [] for category in FALSE_ALARM_CATEGORIES + ["person on ORG", "person on no entity"]}
    for example, spans in zip(examples, coarse):
        text, gold = example["text"], example["gold"]
        for start, end, label in gold:
            if label == "PER":
                entities += 1
                found += any(span[2] == "person" for span in overlaps(start, end, spans))
        for start, end, category in spans:
            touched = sorted({label for _, _, label in overlaps(start, end, gold)})
            if category == "person":
                person_spans += 1
                if "PER" in touched:
                    right += 1
                else:
                    wrong_person_on["+".join(touched) or "no entity"] += 1
                    sample_key = "person on ORG" if touched == ["ORG"] else "person on no entity" if not touched else None
                    if sample_key and len(samples[sample_key]) < SAMPLES_PER_CATEGORY:
                        samples[sample_key].append(f"{text[max(0, start - 40) : start]}[[{text[start:end]}]]{text[end : end + 25]}")
            elif category == "address":
                address_on["+".join(touched) or "no entity"] += 1
            elif category in FALSE_ALARM_CATEGORIES:
                false_alarms[category] += 1
                if len(samples[category]) < SAMPLES_PER_CATEGORY:
                    before = text[max(0, start - 40) : start]
                    samples[category].append(f"{before}[[{text[start:end]}]]{text[end : end + 25]}")
    precision, recall = ratio(right, person_spans), ratio(found, entities)
    return {
        "texts": len(examples),
        "person": {"precision": precision, "recall": recall, "f1": f1(precision, recall), "entities": entities, "spans": person_spans},
        "wrong_person_spans_on": dict(wrong_person_on.most_common()),
        "address_spans_on": dict(address_on.most_common()),
        "false_alarms": {category: false_alarms[category] for category in FALSE_ALARM_CATEGORIES},
        "false_alarm_samples": samples,
    }


def print_tables(results, model_names):
    for dataset_name, by_model in results.items():
        first = by_model[model_names[0]]
        print(f"\n## {dataset_name}: {first['texts']} texts, {first['person']['entities']} person entities\n")
        print("| Model | Person P / R / F1 | Person spans | Wrong person spans | Phone | Email | Account | False alarms per 1,000 texts |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for name in model_names:
            result = by_model[name]
            person, alarms = result["person"], result["false_alarms"]
            wrong = sum(result["wrong_person_spans_on"].values())
            per_thousand = 1000 * sum(alarms.values()) / result["texts"]
            print(
                f"| {name} | {person['precision']:.3f} / {person['recall']:.3f} / {person['f1']:.3f} | {person['spans']} | {wrong} "
                f"| {alarms['phone']} | {alarms['email']} | {alarms['account']} | {per_thousand:.1f} |"
            )
        for title, key in [("Wrong person spans land on", "wrong_person_spans_on"), ("Address spans land on", "address_spans_on")]:
            print(f"\n{title} (gold types touched):\n")
            for name in model_names:
                top = ", ".join(f"{label} {count}" for label, count in list(by_model[name][key].items())[:6])
                print(f"- {name}: {top}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg, help="NAME=PATH_OR_HUB_ID:LABEL_MAP, repeatable")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", default=Path(__file__).parent / "results_real_text.json")
    args = parser.parse_args()

    datasets = {"nemo": load_nemo(), "iahlt_test": load_iahlt_test()}
    results = {name: {} for name in datasets}
    for model_name, model_id, label_map_name in args.model:
        for dataset_name, examples in datasets.items():
            predictions = cached_predictions(dataset_name, model_name, model_id, [e["text"] for e in examples], args.device)
            results[dataset_name][model_name] = check(examples, predictions, LABEL_MAPS[label_map_name])
    print_tables(results, [name for name, _, _ in args.model])
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
