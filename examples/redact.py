"""Replace personal information in a Hebrew text with label placeholders such as [FIRST_NAME].

Run: uv run examples/redact.py [MODEL] < your_text.txt
MODEL is a Hugging Face model id or a local folder (default: CordwainerSmith/GolemPII-v2-base).
Without input on stdin, a built-in sample text is used.
"""

import sys

from transformers import pipeline

SAMPLE = """נושא: RE: ticket #48213 - עדכון פרטי לקוח

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


def redact(text, spans):
    pieces, position = [], 0
    for start, end, label in spans:
        pieces += [text[position:start], f"[{label}]"]
        position = end
    return "".join(pieces) + text[position:]


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "CordwainerSmith/GolemPII-v2-base"
    piped = "" if sys.stdin.isatty() else sys.stdin.read()
    text = piped or SAMPLE

    detector = pipeline("token-classification", model=model, aggregation_strategy="simple", stride=128)
    spans = clean_spans(text, detector(text))
    print(redact(text, spans))
