from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

DEFAULT_MODEL = "CordwainerSmith/GolemPII-v2-base"

GOLEMPII_TO_PRESIDIO = {
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "ID_NUM": "IL_ID_NUMBER",
    "PASSPORT_NUM": "IL_PASSPORT",
    "DRIVER_LICENSE_NUM": "IL_DRIVER_LICENSE",
    "BANK_ACCOUNT_NUM": "IL_BANK_NUMBER",
    "IBAN": "IBAN_CODE",
    "CC_NUM": "CREDIT_CARD",
    "CC_PROVIDER": "CREDIT_CARD_PROVIDER",
    "PHONE_NUM": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "STREET": "STREET_ADDRESS",
    "CITY": "LOCATION",
    "POSTAL_CODE": "IL_POSTAL_CODE",
    "DATE": "DATE_TIME",
}

EDGE_PUNCTUATION = " .,;:!?()[]{}\"'-"


class GolemPIIRecognizer(HuggingFaceNerRecognizer):
    """Hebrew PII recognizer backed by a GolemPII v2 token-classification model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        super().__init__(
            model_name=model_name,
            name="GolemPIIRecognizer",
            supported_language="he",
            label_mapping=GOLEMPII_TO_PRESIDIO,
            **kwargs,
        )

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        results = super().analyze(text, entities, nlp_artifacts)
        return merge_adjacent(text, trim_edges(text, results))


def trim_edges(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    trimmed = []
    for result in results:
        while result.start < result.end and text[result.start] in EDGE_PUNCTUATION:
            result.start += 1
        while result.end > result.start and text[result.end - 1] in EDGE_PUNCTUATION:
            result.end -= 1
        if result.start < result.end:
            trimmed.append(result)
    return trimmed


def merge_adjacent(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    """Join same-type spans separated only by whitespace, so a first and last name become one PERSON."""
    merged = []
    for result in sorted(results, key=lambda r: r.start):
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.entity_type == result.entity_type
            and not text[previous.end : result.start].strip()
        ):
            previous.end = max(previous.end, result.end)
            previous.score = max(previous.score, result.score)
        else:
            merged.append(result)
    return merged


def create_analyzer(model_name: str = DEFAULT_MODEL) -> AnalyzerEngine:
    registry = RecognizerRegistry(supported_languages=["he"])
    registry.add_recognizer(GolemPIIRecognizer(model_name=model_name))
    return AnalyzerEngine(
        nlp_engine=NoOpNlpEngine(models=[{"lang_code": "he", "model_name": "no_op"}]),
        registry=registry,
        supported_languages=["he"],
    )
