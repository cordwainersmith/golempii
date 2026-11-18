import argparse

from presidio_anonymizer import AnonymizerEngine

from golempii_recognizer import DEFAULT_MODEL, create_analyzer

TEXT = """נושא: RE: ticket #48213 - עדכון פרטי לקוח

היי רונית,
הלקוח יואב בן דוד (ת.ז. 034567891) התקשר הבוקר וביקש לעדכן את הכתובת למשלוח: רחוב הרצל 42, רמת גן, מיקוד 5244302.
אפשר להשיג אותו ב-054-7731920 או במייל yoav.bendavid@example.com.
החיוב החודשי ירד מכרטיס ויזה 4580 1234 5678 9015, והזיכוי יועבר לחשבון IL620108000000099999999.
order_id=ORD-2024-88127, סכום: 349.90 ש"ח, תאריך לידה לאימות: 14/03/1987.
[2026-11-12 09:41:07] status=updated

תודה,
Dana Levi | Customer Success"""

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Hugging Face model id or local folder")
args = parser.parse_args()

analyzer = create_analyzer(args.model)
results = analyzer.analyze(text=TEXT, language="he")

for result in sorted(results, key=lambda r: r.start):
    print(f"{result.entity_type:<22} {result.score:.2f}  {TEXT[result.start:result.end]}")
print()
print(AnonymizerEngine().anonymize(text=TEXT, analyzer_results=results).text)
