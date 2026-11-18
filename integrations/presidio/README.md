# GolemPII v2 for Microsoft Presidio

Hebrew PII detection and anonymization in Presidio, using the GolemPII v2 models:
[base](https://huggingface.co/CordwainerSmith/GolemPII-v2-base),
[small](https://huggingface.co/CordwainerSmith/GolemPII-v2-small),
[tiny](https://huggingface.co/CordwainerSmith/GolemPII-v2-tiny).
Try them in the browser first: [GolemPII v2 demo](https://huggingface.co/spaces/CordwainerSmith/GolemPII-v2-demo).

The models read Hebrew and the English mixed into it (emails, tickets, chat, logs, JSON), so one analyzer with `language="he"` covers both.

## How it works

Presidio and spaCy ship nothing for Hebrew. This recipe uses two pieces that Presidio already provides for this case:

- `HuggingFaceNerRecognizer`, which runs a Hugging Face token-classification model directly on the text and returns character offsets, with no spaCy tokenizer in between. Hebrew glues prefixes to words (`ב-054...`, `לרונית`), so aligning model spans to spaCy tokens would stretch them.
- `NoOpNlpEngine`, which registers language `he` without loading any spaCy model.

`golempii_recognizer.py` subclasses the recognizer to add the label mapping, trim punctuation from span edges, and merge adjacent first and last names into one `PERSON`. `create_analyzer()` wires it into an `AnalyzerEngine`.

## Install

Python 3.11 to 3.14. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

`requirements.txt` pins every package; `requirements.in` lists the four direct ones.

## Quick start

Copy `golempii_recognizer.py` next to your code:

```python
from presidio_anonymizer import AnonymizerEngine
from golempii_recognizer import create_analyzer

analyzer = create_analyzer("CordwainerSmith/GolemPII-v2-base")
text = "היי רונית, הלקוח יואב בן דוד (ת.ז. 034567891) זמין ב-054-7731920, הזמנה ORD-2024-88127."
results = analyzer.analyze(text=text, language="he")
print(AnonymizerEngine().anonymize(text=text, analyzer_results=results).text)
```

```text
היי <PERSON>, הלקוח <PERSON> (ת.ז. <IL_ID_NUMBER>) זמין ב-<PHONE_NUMBER>, הזמנה ORD-2024-88127.
```

To use your own registry, add `GolemPIIRecognizer(model_name=...)` to it; the analyzer's NLP engine must list `he` as a language. The first call downloads the model from Hugging Face; a local folder path works too.

## Label mapping

| GolemPII label | Presidio entity |
|---|---|
| `FIRST_NAME`, `LAST_NAME` | `PERSON` (adjacent spans merged) |
| `ID_NUM` | `IL_ID_NUMBER` |
| `PASSPORT_NUM` | `IL_PASSPORT` |
| `DRIVER_LICENSE_NUM` | `IL_DRIVER_LICENSE` |
| `BANK_ACCOUNT_NUM` | `IL_BANK_NUMBER` |
| `IBAN` | `IBAN_CODE` |
| `CC_NUM` | `CREDIT_CARD` |
| `CC_PROVIDER` | `CREDIT_CARD_PROVIDER` |
| `PHONE_NUM` | `PHONE_NUMBER` |
| `EMAIL` | `EMAIL_ADDRESS` |
| `STREET` | `STREET_ADDRESS` |
| `CITY` | `LOCATION` |
| `POSTAL_CODE` | `IL_POSTAL_CODE` |
| `DATE` | `DATE_TIME` |

Standard Presidio names are used where one exists; the `IL_` names follow Presidio's country-prefix convention (`UK_PASSPORT`, `US_DRIVER_LICENSE`). Edit `GOLEMPII_TO_PRESIDIO` to change them.

## Which model size

| Model | Parameters | Choose it for |
|---|---:|---|
| [GolemPII-v2-base](https://huggingface.co/CordwainerSmith/GolemPII-v2-base) | 277M | Hebrew mixed with English in enterprise text, and the best recall on identity, card and bank numbers. The default. |
| [GolemPII-v2-small](https://huggingface.co/CordwainerSmith/GolemPII-v2-small) | 125M | Hebrew prose: best at telling person names from organisations, at less than half the size of base and the same speed |
| [GolemPII-v2-tiny](https://huggingface.co/CordwainerSmith/GolemPII-v2-tiny) | 45M | Speed or size first: about five times faster, with lower accuracy on enterprise text |

## Example output

`example.py` analyzes and anonymizes the sample from the model card, with one log line added to show that a timestamp is left alone:

```bash
.venv/bin/python example.py                                   # GolemPII-v2-base from Hugging Face
.venv/bin/python example.py CordwainerSmith/GolemPII-v2-tiny  # any model id or local folder
```

Real output with GolemPII-v2-base. GolemPII-v2-small and GolemPII-v2-tiny produced the same spans and the same anonymized text on this sample (scores 0.996 to 1.000, shown rounded):

```text
PERSON                 1.00  רונית
PERSON                 1.00  יואב בן דוד
IL_ID_NUMBER           1.00  034567891
STREET_ADDRESS         1.00  רחוב הרצל 42
LOCATION               1.00  רמת גן
IL_POSTAL_CODE         1.00  5244302
PHONE_NUMBER           1.00  054-7731920
EMAIL_ADDRESS          1.00  yoav.bendavid@example.com
CREDIT_CARD_PROVIDER   1.00  ויזה
CREDIT_CARD            1.00  4580 1234 5678 9015
IBAN_CODE              1.00  IL620108000000099999999
DATE_TIME              1.00  14/03/1987
PERSON                 1.00  Dana Levi

נושא: RE: ticket #48213 - עדכון פרטי לקוח

היי <PERSON>,
הלקוח <PERSON> (ת.ז. <IL_ID_NUMBER>) התקשר הבוקר וביקש לעדכן את הכתובת למשלוח: <STREET_ADDRESS>, <LOCATION>, מיקוד <IL_POSTAL_CODE>.
אפשר להשיג אותו ב-<PHONE_NUMBER> או במייל <EMAIL_ADDRESS>.
החיוב החודשי ירד מכרטיס <CREDIT_CARD_PROVIDER> <CREDIT_CARD>, והזיכוי יועבר לחשבון <IBAN_CODE>.
order_id=ORD-2024-88127, סכום: 349.90 ש"ח, תאריך לידה לאימות: <DATE_TIME>.
[2026-11-12 09:41:07] status=updated

תודה,
<PERSON> | Customer Success
```

The ticket number, order id, amount and timestamp are left alone.

## Limits

- The models assist redaction; they do not guarantee anonymization. Text that must be anonymized needs human review.
- Not labelled, by design: organisations, job titles, countries, and reference numbers such as order, ticket and employee numbers.
- Weak spots stated on the model cards: dates (`DATE` F1 0.530 on the REDACT benchmark), driver licence and card numbers, organisations named after people, the last four digits of a card ("ending 4821"), identity numbers glued to other text, and obfuscated emails ("name AT domain DOT com").
- Arabic and English-only text were not evaluated.
- Long texts are split into overlapping 400-character chunks by Presidio's default chunker. Pass `chunk_size` and `chunk_overlap` to `GolemPIIRecognizer` to change that.
- `NoOpNlpEngine` produces no tokens or lemmas, so Presidio's context words from the text do not boost scores. Context passed to `analyze(context=...)` still works.
- Adjacent spans of the same type separated only by whitespace are merged, so two names written back to back with no punctuation become one `PERSON`.
- Runs on CPU or CUDA. Presidio's recognizer does not use Apple's MPS.
