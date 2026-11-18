![GolemPII v2: Hebrew PII detection](banner.png)

# GolemPII

GolemPII finds personal information in Hebrew text, including Hebrew mixed with English as it appears in enterprise systems: emails, tickets, chat, key-value records, JSON and logs. It is a small token classifier with 15 labels that runs locally, on a CPU if needed, so you can place it in front of LLM calls, logs or analytics and replace names, identity numbers, phones, cards and addresses before the text leaves your machine.

[Try the demo](https://huggingface.co/spaces/CordwainerSmith/GolemPII-v2-demo) | [Read the write-up](https://liranbaba.dev/blog/agent-built-hebrew-pii-model/) | [Models on Hugging Face](https://huggingface.co/CordwainerSmith/GolemPII-v2-base)

## Models

| Model | Encoder | Parameters | Licence | Choose it for |
|---|---|---:|---|---|
| [GolemPII-v2-base](https://huggingface.co/CordwainerSmith/GolemPII-v2-base) | `xlm-roberta-base` | 277M | MIT | Hebrew mixed with English in enterprise text (emails, tickets, logs, JSON), and the best recall of the three on identity, card and bank numbers |
| [GolemPII-v2-small](https://huggingface.co/CordwainerSmith/GolemPII-v2-small) | `HalleluBERT_base` | 125M | MIT | Hebrew prose: the best of the three at telling person names from organisations, at less than half the size of base and the same speed |
| [GolemPII-v2-tiny](https://huggingface.co/CordwainerSmith/GolemPII-v2-tiny) | `dictabert-tiny` | 45M | CC-BY-4.0 | When speed or size matters most (about five times faster), with lower accuracy on enterprise text |

If you are unsure, start with GolemPII-v2-base.

## Quick start

With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/cordwainersmith/golempii.git
cd golempii
uv run examples/quickstart.py
```

The first run downloads the model from Hugging Face. Pass another model id or a local folder as the argument, for example `uv run examples/quickstart.py CordwainerSmith/GolemPII-v2-tiny`.

The core of it is a standard `transformers` pipeline:

```python
from transformers import pipeline

detector = pipeline(
    "token-classification",
    model="CordwainerSmith/GolemPII-v2-base",
    aggregation_strategy="simple",
    stride=128,  # lets the pipeline handle texts longer than 512 tokens
)
entities = detector(text)
```

[`examples/quickstart.py`](examples/quickstart.py) runs it on this synthetic support ticket:

```text
נושא: RE: ticket #48213 - עדכון פרטי לקוח

היי רונית,
הלקוח יואב בן דוד (ת.ז. 034567891) התקשר הבוקר וביקש לעדכן את הכתובת למשלוח: רחוב הרצל 42, רמת גן, מיקוד 5244302.
אפשר להשיג אותו ב-054-7731920 או במייל yoav.bendavid@example.com.
החיוב החודשי ירד מכרטיס ויזה 4580 1234 5678 9015, והזיכוי יועבר לחשבון IL620108000000099999999.
order_id=ORD-2024-88127, סכום: 349.90 ש"ח, תאריך לידה לאימות: 14/03/1987.

תודה,
Dana Levi | Customer Success
```

and prints the detected spans:

```text
FIRST_NAME   רונית
FIRST_NAME   יואב
LAST_NAME    בן דוד
ID_NUM       034567891
STREET       רחוב הרצל 42
CITY         רמת גן
POSTAL_CODE  5244302
PHONE_NUM    054-7731920
EMAIL        yoav.bendavid@example.com
CC_PROVIDER  ויזה
CC_NUM       4580 1234 5678 9015
IBAN         IL620108000000099999999
DATE         14/03/1987
FIRST_NAME   Dana
LAST_NAME    Levi
```

The script passes the pipeline output through `clean_spans`, a short function that trims punctuation from span edges and joins pieces of one span. The pipeline groups sub-word tokens, so a raw span can carry a bracket, a hyphen or a full stop at its edge, or arrive in two pieces. `clean_spans` is optional; the scores below were measured on the raw pipeline output, without it.

## Redact

[`examples/redact.py`](examples/redact.py) replaces every span with its label:

```python
def redact(text, spans):
    pieces, position = [], 0
    for start, end, label in spans:
        pieces += [text[position:start], f"[{label}]"]
        position = end
    return "".join(pieces) + text[position:]
```

Run it on the sample ticket, or pipe in your own text:

```bash
uv run examples/redact.py
uv run examples/redact.py CordwainerSmith/GolemPII-v2-base < my_text.txt
```

Output for the sample ticket. The ticket number, the order id and the amount are left alone:

```text
נושא: RE: ticket #48213 - עדכון פרטי לקוח

היי [FIRST_NAME],
הלקוח [FIRST_NAME] [LAST_NAME] (ת.ז. [ID_NUM]) התקשר הבוקר וביקש לעדכן את הכתובת למשלוח: [STREET], [CITY], מיקוד [POSTAL_CODE].
אפשר להשיג אותו ב-[PHONE_NUM] או במייל [EMAIL].
החיוב החודשי ירד מכרטיס [CC_PROVIDER] [CC_NUM], והזיכוי יועבר לחשבון [IBAN].
order_id=ORD-2024-88127, סכום: 349.90 ש"ח, תאריך לידה לאימות: [DATE].

תודה,
[FIRST_NAME] [LAST_NAME] | Customer Success
```

All three models produce the same spans and the same redacted text for this sample.

## Integrations

**PLACEHOLDER (to be replaced): Presidio integration in `integrations/presidio/`.**

## Labels

| Group | Labels |
|---|---|
| Person | `FIRST_NAME`, `LAST_NAME` |
| Identity documents | `ID_NUM`, `PASSPORT_NUM`, `DRIVER_LICENSE_NUM` |
| Financial | `BANK_ACCOUNT_NUM`, `IBAN`, `CC_NUM`, `CC_PROVIDER` |
| Contact | `PHONE_NUM`, `EMAIL` |
| Address | `STREET`, `CITY`, `POSTAL_CODE` |
| Other | `DATE` |

Masked card numbers (`**** **** **** 1234`) are labelled `CC_NUM`. Organisations, job titles, countries, and reference numbers such as order, ticket and employee numbers are not labelled.

## Results

All numbers are from text the models were not trained on. REDACT is an external benchmark of enterprise-style Hebrew text. Its Hebrew part was split once into two halves: `select`, used for every decision during development, and `final`, scored once after the model was frozen. The numbers here are from `final` (311 records, 4,936 entities).

Character level counts every character of a span. Entity level counts an entity as found when a span of the right category touches it, so it does not depend on where span boundaries are drawn. REDACT has five categories; the 15 labels are mapped onto them, and a `CITY` span on its own is not counted as an address.

### REDACT `final`, five categories

| Model | Parameters | Character P / R / F1 | Entity P / R / F1 |
|---|---:|---|---|
| GolemPII-v1 | 277M | 0.592 / 0.884 / 0.709 | 0.601 / 0.913 / 0.725 |
| GolemPII-v2-base | 277M | 0.886 / 0.951 / 0.917 | 0.872 / 0.983 / 0.924 |
| GolemPII-v2-small | 125M | 0.890 / 0.943 / 0.916 | 0.867 / 0.974 / 0.917 |
| GolemPII-v2-tiny | 45M | 0.844 / 0.927 / 0.884 | 0.797 / 0.968 / 0.874 |

### REDACT `final`, by category (entity level, P / R / F1)

| Category | GolemPII-v1 | GolemPII-v2-base | GolemPII-v2-small | GolemPII-v2-tiny |
|---|---|---|---|---|
| Person | 0.804 / 0.900 / 0.849 | 0.888 / 0.986 / 0.934 | 0.885 / 0.981 / 0.930 | 0.846 / 0.971 / 0.904 |
| Address | 0.597 / 1.000 / 0.747 | 0.777 / 1.000 / 0.874 | 0.781 / 1.000 / 0.877 | 0.735 / 0.985 / 0.842 |
| Phone | 0.238 / 0.969 / 0.383 | 0.940 / 0.993 / 0.966 | 0.927 / 0.993 / 0.959 | 0.883 / 0.965 / 0.922 |
| Email | 0.763 / 0.996 / 0.864 | 0.940 / 0.998 / 0.968 | 0.934 / 0.998 / 0.965 | 0.734 / 1.000 / 0.846 |
| Account (identity, card and bank numbers) | 0.265 / 0.844 / 0.403 | 0.762 / 0.905 / 0.827 | 0.723 / 0.820 / 0.768 | 0.571 / 0.890 / 0.696 |

### Person names on real text (NEMO)

NEMO is a human-labelled corpus of Hebrew news sentences (6,143 sentences, 2,588 person mentions). It was never used for training, and it was read once, after the model was chosen.

| Model | Person P / R / F1 |
|---|---|
| GolemPII-v1 | 0.541 / 0.824 / 0.653 |
| GolemPII-v2-base | 0.853 / 0.925 / 0.887 |
| GolemPII-v2-small | 0.928 / 0.956 / 0.942 |
| GolemPII-v2-tiny | 0.856 / 0.945 / 0.899 |

### Speed and size

Measured on an Apple M4 Pro (48 GB, macOS 27.0.1, torch 2.15.0, transformers 5.22.1) over 1,000 synthetic enterprise-style texts with a mean length of 1,512 characters, with the pipeline settings shown above and batch size 8. Median of three runs; other programs were running, so treat differences of a few percent as noise.

| Model | Parameters | File size | Seconds per 1,000 texts, Apple GPU (MPS) | Seconds per 1,000 texts, CPU |
|---|---:|---:|---:|---:|
| GolemPII-v2-base | 277M | 1,110 MB | 63.8 | 115.6 |
| GolemPII-v2-small | 125M | 502 MB | 65.8 | 120.9 |
| GolemPII-v2-tiny | 45M | 179 MB | 13.0 | 15.6 |

The per-label scores, code-switching results and training details are on the [model cards](https://huggingface.co/CordwainerSmith/GolemPII-v2-base). The [`evaluation/`](evaluation/) folder holds the scoring code used for every number on this page.

## Limits

- **Organisations named like people.** On news text the model tags some sports clubs, companies and firms named after their founders as people. This is its main error on NEMO, and the amount differs between training runs.
- **Account numbers:** 0.905 of the identity, card and bank numbers in REDACT are found. Reference numbers such as customer and student numbers are not labelled by this model, by design. Known misses: references to the last four digits of a card ("ending 4821"), tax file numbers, passport numbers inside a machine-readable line, and identity numbers glued to other text. Obfuscated emails ("name AT domain DOT com") are missed.
- **Driver licence and card numbers** score low on REDACT (0.411 and 0.671 character F1, on 13 and 53 entities).
- **Dates:** `DATE` scores 0.530 on REDACT, whose date labelling differs from ours; treat date detection as weak.
- **Span boundaries:** address spans are often shorter than REDACT's (the street name without the word "street"), which lowers character-level address scores while the entities are still found.
- **Only names are tested on real text.** Numeric PII in real enterprise text is tested only on REDACT, which is itself synthetic, and on our own held-out templates.

The figures above are for GolemPII-v2-base. All of these limits apply to the smaller models too, and in addition:

- **GolemPII-v2-small** finds fewer identity, card and bank numbers than the base model (entity recall 0.820 against 0.905). Organisations named like people are still its most common name error on news text (93 of the 2,589 person spans it predicts on NEMO), though far less often than for the base model.
- **GolemPII-v2-tiny** has clearly lower precision on account numbers and emails than the base model: it tags more reference numbers as identity or card numbers, and more non-email strings as emails. It comes from one training run; the spread between runs was not measured.

Out of scope or not evaluated:

- **Other languages.** Arabic, and English text on its own, were not evaluated.
- **Other sources.** Speech transcripts, OCR output and handwriting were not evaluated.
- **Anonymisation.** The model is a detector that assists redaction, not a guarantee of anonymisation or of compliance with any privacy law. Text that must be anonymised needs human review.
- **Other entity types.** Organisations, job titles, countries and reference numbers (order, ticket and employee numbers) are not labelled.

## Report a miss

If a model misses personal information or tags something that is not, [open an issue](https://github.com/cordwainersmith/golempii/issues/new/choose) with the text you ran, the model, what it returned and what you expected.

**Use synthetic text only. Never paste real personal data into an issue.** Invent names, numbers and addresses that look like the real thing.

## Citation

If you use GolemPII in a product, a paper or a derived model, please credit it as **GolemPII-v2-base by [Liran Baba](https://liranbaba.dev)** (or the variant you use), with a link to the model on Hugging Face.

```bibtex
@misc{baba2026golempii_v2_base,
  author       = {Baba, Liran},
  title        = {GolemPII v2 base: personal information detection for Hebrew text},
  year         = {2026},
  version      = {2.0},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/CordwainerSmith/GolemPII-v2-base}}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff).

## Licence

The code in this repository is MIT, copyright (c) 2026 Liran Baba; see [`LICENSE`](LICENSE).

The models are licensed separately: GolemPII-v2-base and GolemPII-v2-small are MIT, GolemPII-v2-tiny is CC-BY-4.0, following its encoder `dicta-il/dictabert-tiny` by Dicta.
