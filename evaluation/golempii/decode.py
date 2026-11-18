"""Word-level span decoding and structural validators for token-classification output."""

import re
import unicodedata
from bisect import bisect_right

HEBREW_PREFIXES = set("לבומשהכ")
NAME_LABELS = {"FIRST_NAME", "LAST_NAME"}
NUMBER_LABELS = {"ID_NUM", "BANK_ACCOUNT_NUM", "PASSPORT_NUM", "DRIVER_LICENSE_NUM"}

ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?")
SHORT_DATE = re.compile(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")
CLOCK_TIME = re.compile(r"\d{1,2}:\d{2}(:\d{2})?")
IP_ADDRESS = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
HEX_STRING = re.compile(r"[0-9a-fA-F]{8,}")


def normalize_label(label):
    if label == "O":
        return None
    if label[:2] in ("B-", "I-"):
        return label[2:]
    return label


def is_punctuation(char):
    return unicodedata.category(char)[0] in "PS"


def trim_punctuation(text, start, end):
    # A leading plus belongs to international phone numbers.
    while start < end and is_punctuation(text[start]) and text[start] != "+":
        start += 1
    while start < end and is_punctuation(text[end - 1]):
        end -= 1
    return start, end


def word_label(token_labels):
    # Majority of the labelled sub-tokens, unlabelled ones do not vote. Ties go to the label seen first.
    labelled = [label for label in token_labels if label is not None]
    if not labelled:
        return None
    return max(dict.fromkeys(labelled), key=labelled.count)


def decode_spans(text, offsets, labels):
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    word_starts = [start for start, _ in words]
    tokens_by_word = [[] for _ in words]
    for (start, end), label in zip(offsets, labels):
        while start < end and text[start].isspace():
            start += 1
        while start < end and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        word_index = bisect_right(word_starts, start) - 1
        tokens_by_word[word_index].append((start, end, normalize_label(label)))

    decoded_words = []
    for (word_start, word_end), tokens in zip(words, tokens_by_word):
        label = word_label([token_label for _, _, token_label in tokens])
        if label is None:
            continue
        first_start, first_end, first_label = tokens[0]
        if first_label is None and first_end - first_start == 1 and text[first_start] in HEBREW_PREFIXES:
            word_start = first_end
        start, end = trim_punctuation(text, word_start, word_end)
        if start < end:
            decoded_words.append((start, end, label))

    spans = []
    for start, end, label in decoded_words:
        if spans and spans[-1][2] == label and text[spans[-1][1]:start].isspace():
            spans[-1] = (spans[-1][0], end, label)
        else:
            spans.append((start, end, label))
    return spans


def digits_of(value):
    return re.sub(r"\D", "", value)


def luhn_ok(digits):
    if not digits.isdigit():
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def israeli_id_ok(digits):
    if not digits.isdigit() or len(digits) > 9:
        return False
    total = 0
    for position, char in enumerate(digits.zfill(9)):
        product = int(char) * (1 + position % 2)
        total += product - 9 if product > 9 else product
    return total % 10 == 0


def iban_ok(value):
    compact = re.sub(r"[\s-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(char, 36)) for char in rearranged)
    return int(number) % 97 == 1


def is_valid(label, value):
    value = value.strip()
    digit_count = len(digits_of(value))
    if label == "CC_NUM":
        return 13 <= digit_count <= 19 and luhn_ok(digits_of(value))
    if label == "PHONE_NUM":
        looks_like_time = any(
            pattern.fullmatch(value) for pattern in (ISO_TIMESTAMP, SHORT_DATE, CLOCK_TIME)
        )
        return 7 <= digit_count <= 15 and not looks_like_time
    if label == "IBAN":
        return iban_ok(value) if re.match(r"[A-Za-z]{2}", value) else True
    if label in NUMBER_LABELS:
        looks_like_hash = HEX_STRING.fullmatch(value) and re.search(r"[a-fA-F]", value)
        return digit_count >= 5 and not (
            ISO_TIMESTAMP.fullmatch(value) or looks_like_hash or IP_ADDRESS.fullmatch(value)
        )
    if label in NAME_LABELS:
        return len(value) >= 2 and not any(char.isdigit() for char in value)
    return True
