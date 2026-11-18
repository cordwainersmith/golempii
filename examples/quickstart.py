"""Find personal information in a Hebrew text with GolemPII and print the detected spans.

Run: uv run examples/quickstart.py [MODEL]
MODEL is a Hugging Face model id or a local folder (default: CordwainerSmith/GolemPII-v2-base).
"""

import sys

from transformers import pipeline

MODEL = sys.argv[1] if len(sys.argv) > 1 else "CordwainerSmith/GolemPII-v2-base"

TEXT = """נושא: RE: ticket #48213 - עדכון פרטי לקוח

היי רונית,
הלקוח יואב בן דוד (ת.ז. 034567891) התקשר הבוקר וביקש לעדכן את הכתובת למשלוח: רחוב הרצל 42, רמת גן, מיקוד 5244302.
אפשר להשיג אותו ב-054-7731920 או במייל yoav.bendavid@example.com.
החיוב החודשי ירד מכרטיס ויזה 4580 1234 5678 9015, והזיכוי יועבר לחשבון IL620108000000099999999.
order_id=ORD-2024-88127, סכום: 349.90 ש"ח, תאריך לידה לאימות: 14/03/1987.

תודה,
Dana Levi | Customer Success"""

EDGE_PUNCTUATION = " .,;:!?()[]{}\"'-"


def clean_spans(text, entities):
    """Trim punctuation from span edges and join spans of one label that touch or overlap."""
    spans = []
    for entity in entities:
        start, end, label = entity["start"], entity["end"], entity["entity_group"]
        while start < end and text[start] in EDGE_PUNCTUATION:
            start += 1
        while end > start and text[end - 1] in EDGE_PUNCTUATION:
            end -= 1
        if start == end:
            continue
        if spans and spans[-1][2] == label and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]), label)
        else:
            spans.append((start, end, label))
    return spans


detector = pipeline(
    "token-classification",
    model=MODEL,
    aggregation_strategy="simple",
    stride=128,  # lets the pipeline handle texts longer than 512 tokens
)

for start, end, label in clean_spans(TEXT, detector(TEXT)):
    print(f"{label:<12} {TEXT[start:end]}")
