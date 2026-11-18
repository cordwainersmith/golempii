import random
import re
import string
import uuid
from collections.abc import Callable
from pathlib import Path

LEXICON_DIR = Path(__file__).parent / "lexicons"
LATIN_SHARE = 0.15
FOREIGN_PASSPORT_SHARE = 0.25


def load_lexicon(name: str) -> list[tuple[str, str]]:
    seen = {}
    for line in (LEXICON_DIR / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            hebrew, latin = line.split(",", 1)
            seen.setdefault(hebrew, latin)
    return list(seen.items())


FIRST_NAMES = load_lexicon("first_names_he.txt")
LAST_NAMES = load_lexicon("last_names_he.txt")
CITIES = load_lexicon("cities_he.txt")
STREETS = load_lexicon("streets_he.txt")
LATIN_FIRST = dict(FIRST_NAMES)
LATIN_LAST = dict(LAST_NAMES)


def digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def pick_name(rng: random.Random, lexicon: list[tuple[str, str]]) -> str:
    hebrew, latin = rng.choice(lexicon)
    return latin if rng.random() < LATIN_SHARE else hebrew


def first_name(rng: random.Random) -> str:
    return pick_name(rng, FIRST_NAMES)


def last_name(rng: random.Random) -> str:
    return pick_name(rng, LAST_NAMES)


def israeli_id_check_digit(first_eight: str) -> str:
    total = 0
    for position, char in enumerate(first_eight):
        product = int(char) * (1 + position % 2)
        total += product - 9 if product > 9 else product
    return str((10 - total % 10) % 10)


def is_valid_israeli_id(value: str) -> bool:
    value = value.zfill(9)
    return len(value) == 9 and value.isdigit() and israeli_id_check_digit(value[:8]) == value[8]


def israeli_id(rng: random.Random) -> str:
    leading_zero = rng.random() < 0.15
    body = ("0" if leading_zero else str(rng.randint(1, 9))) + digits(rng, 7)
    full = body + israeli_id_check_digit(body)
    style = rng.random()
    if style < 0.65:
        return full
    if style < 0.78:
        return f"{full[:8]}-{full[8]}"
    if style < 0.88:
        return f"{full[:3]} {full[3:6]} {full[6:]}"
    return full.lstrip("0") if leading_zero else full


MOBILE_AREAS = ["50", "52", "53", "54", "55", "58"]
LANDLINE_AREAS = ["2", "3", "4", "8", "9"]
VOIP_AREAS = ["72", "73", "74", "76", "77", "78"]


def phone(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.65:
        area = rng.choice(MOBILE_AREAS)
    elif kind < 0.9:
        area = rng.choice(LANDLINE_AREAS)
    else:
        area = rng.choice(VOIP_AREAS)
    subscriber = str(rng.randint(2, 9)) + digits(rng, 6)
    head, tail = subscriber[:3], subscriber[3:]
    if rng.random() < 0.72:
        local = "0" + area
        forms = [
            f"{local}{subscriber}",
            f"{local}-{subscriber}",
            f"{local}-{head}-{tail}",
            f"{local} {head} {tail}",
            f"{local} {subscriber}",
            f"({local}) {head}-{tail}",
            f"({local}) {subscriber}",
        ]
    else:
        forms = [
            f"+972{area}{subscriber}",
            f"+972-{area}-{head}-{tail}",
            f"+972 {area} {head} {tail}",
            f"+972-{area}-{subscriber}",
            f"972{area}{subscriber}",
            f"972-{area}-{subscriber}",
            f"+972 (0){area} {subscriber}",
        ]
    return rng.choice(forms)


BANK_CODES = ["10", "11", "12", "13", "14", "17", "20", "31", "46", "52", "54", "68"]


def bank_account(rng: random.Random) -> str:
    bank = rng.choice(BANK_CODES)
    branch = str(rng.randint(100, 999))
    account = digits(rng, rng.choice([6, 6, 7, 8, 9]))
    forms = [
        f"{bank}-{branch}-{account}",
        f"{bank}-{branch}-{account}",
        f"{bank}/{branch}/{account}",
        f"{branch}/{account}",
        f"{branch}-{account}",
        f"{bank} {branch} {account}",
        account,
        account,
        f"{bank}{branch}{account}",
    ]
    return rng.choice(forms)


IBAN_BANK_CODES = ["010", "011", "012", "013", "014", "017", "020", "031", "046", "052", "054", "068"]


def iban_check_digits(country: str, bban: str) -> str:
    rearranged = bban + country + "00"
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return f"{98 - int(numeric) % 97:02d}"


def is_valid_iban(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def iban(rng: random.Random) -> str:
    bban = rng.choice(IBAN_BANK_CODES) + str(rng.randint(100, 999)) + digits(rng, 13)
    full = "IL" + iban_check_digits("IL", bban) + bban
    if rng.random() < 0.5:
        return full
    return " ".join(full[i : i + 4] for i in range(0, len(full), 4))


def luhn_check_digit(partial: str) -> str:
    total = 0
    for position, char in enumerate(reversed(partial)):
        value = int(char)
        if position % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - total % 10) % 10)


def is_luhn_valid(value: str) -> bool:
    return value.isdigit() and luhn_check_digit(value[:-1]) == value[-1]


def luhn_number(rng: random.Random, prefix: str, length: int) -> str:
    body = prefix + digits(rng, length - len(prefix) - 1)
    return body + luhn_check_digit(body)


def isracard_number(rng: random.Random) -> str:
    # Weighted sum of the 9 digits (weights 9 down to 1) must be divisible by 11; 8-digit numbers are zero padded.
    while True:
        body = ("0" if rng.random() < 0.4 else str(rng.randint(1, 9))) + digits(rng, 7)
        total = sum(int(char) * (9 - position) for position, char in enumerate(body))
        last = -total % 11
        if last <= 9:
            number = body + str(last)
            return number[1:] if number[0] == "0" else number


def is_isracard_valid(value: str) -> bool:
    if len(value) not in (8, 9) or not value.isdigit():
        return False
    padded = value.zfill(9)
    return sum(int(char) * (9 - position) for position, char in enumerate(padded)) % 11 == 0


CARD_PROVIDER_NAMES = {
    "visa": ["ויזה", "ויזה", "ויזה כאל", "Visa", "VISA"],
    "mastercard": ["מאסטרקארד", "מאסטרקרד", "מסטרקארד", "Mastercard", "MasterCard"],
    "amex": ["אמריקן אקספרס", "אמריקן אקספרס", "אמקס", "American Express", "Amex"],
    "isracard": ["ישראכרט", "ישראכרט", "Isracard"],
    "diners": ["דיינרס", "דיינרס קלאב", "Diners Club", "Diners"],
}
CARD_PROVIDER_WEIGHTS = {"visa": 4, "mastercard": 4, "amex": 1, "isracard": 3, "diners": 1}


def pick_card_provider(rng: random.Random) -> str:
    keys = list(CARD_PROVIDER_WEIGHTS)
    return rng.choices(keys, weights=[CARD_PROVIDER_WEIGHTS[key] for key in keys])[0]


def card_number(rng: random.Random, provider: str) -> str:
    if provider == "isracard":
        return isracard_number(rng)
    if provider == "visa":
        raw = luhn_number(rng, "4", 16)
        groups = [4, 4, 4, 4]
    elif provider == "mastercard":
        prefix = rng.choice(["51", "52", "53", "54", "55", "2221", "2500", "2720"])
        raw = luhn_number(rng, prefix, 16)
        groups = [4, 4, 4, 4]
    elif provider == "amex":
        raw = luhn_number(rng, rng.choice(["34", "37"]), 15)
        groups = [4, 6, 5]
    else:
        raw = luhn_number(rng, rng.choice(["36", "300", "301", "305"]), 14)
        groups = [4, 6, 4]
    style = rng.random()
    if style < 0.35:
        return raw
    parts = []
    start = 0
    for size in groups:
        parts.append(raw[start : start + size])
        start += size
    return ("-" if style < 0.6 else " ").join(parts)


def masked_card_number(rng: random.Random, provider: str) -> str:
    last_four = re.sub(r"\D", "", card_number(rng, provider))[-4:]
    return rng.choice(
        [
            f"**** **** **** {last_four}",
            f"****-****-****-{last_four}",
            f"XXXX-XXXX-XXXX-{last_four}",
            f"**** {last_four}",
            f"****{last_four}",
            f"xxxx{last_four}",
        ]
    )


def card_ending_reference(rng: random.Random, provider: str) -> str:
    last_four = re.sub(r"\D", "", card_number(rng, provider))[-4:]
    if rng.random() < 0.4:
        return rng.choice([f"המסתיים ב-{last_four}", f"מסתיים ב-{last_four}", f"שמסתיים ב-{last_four}"])
    latin_names = [name for name in CARD_PROVIDER_NAMES[provider] if name.isascii()]
    name = rng.choice(latin_names + [latin_names[0].upper()])
    return rng.choice(
        [
            f"{name} ending {last_four}",
            f"{name} ending in {last_four}",
            f"{name.lower().replace(' ', '_')}_ending_{last_four}",
            f"ending in {last_four}",
            f"ending {last_four}",
        ]
    )


def card_provider_name(rng: random.Random, provider: str) -> str:
    return rng.choice(CARD_PROVIDER_NAMES[provider])


def credit_card(rng: random.Random) -> str:
    return card_number(rng, pick_card_provider(rng))


def cc_provider(rng: random.Random) -> str:
    return card_provider_name(rng, pick_card_provider(rng))


def passport_number(rng: random.Random) -> str:
    if rng.random() < FOREIGN_PASSPORT_SHARE:
        letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice([1, 2])))
        return letters + digits(rng, 7)
    return str(rng.randint(1, 9)) + digits(rng, rng.choice([7, 8]))


def driver_license(rng: random.Random) -> str:
    length = rng.choices([7, 8, 9], weights=[4, 4, 2])[0]
    return str(rng.randint(1, 9)) + digits(rng, length - 1)


def house_number(rng: random.Random) -> str:
    number = str(min(int(rng.expovariate(1 / 30)) + 1, 250))
    if rng.random() < 0.05:
        number += rng.choice(["א", "ב", "ג"])
    return number


def street(rng: random.Random) -> str:
    name, latin = rng.choice(STREETS)
    number = house_number(rng)
    style = rng.random()
    if style < 0.08:
        if rng.random() < 0.5:
            return f"{latin} {number}"
        return f"{number} {latin} St."
    is_boulevard = name.startswith(("שדרות", "דרך"))
    if style < 0.58:
        return f"{name} {number}"
    if style < 0.70:
        apartment = rng.randint(1, 60)
        return rng.choice(
            [f"{name} {number} דירה {apartment}", f"{name} {number}, דירה {apartment}", f"{name} {number}/{apartment}"]
        )
    if is_boulevard:
        return f"{name} {number}"
    if style < 0.82:
        return f"שד' {name} {number}"
    if style < 0.88:
        return f"שדרות {name} {number}"
    return f"רחוב {name} {number}"


def city(rng: random.Random) -> str:
    name, latin = rng.choice(CITIES)
    return latin if rng.random() < 0.08 else name


def postal_code(rng: random.Random) -> str:
    if rng.random() < 0.07:
        return str(rng.randint(1, 9)) + digits(rng, 4)
    return str(rng.randint(1, 9)) + digits(rng, 6)


EMAIL_DOMAINS = [
    "gmail.com",
    "gmail.com",
    "gmail.com",
    "walla.co.il",
    "walla.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "012.net.il",
    "bezeqint.net",
    "netvision.net.il",
    "icloud.com",
    "proton.me",
    "mail.huji.ac.il",
    "post.tau.ac.il",
    "technion.ac.il",
    "gov.il",
    "co.il",
]
COMPANY_WORDS = [
    "nexora", "datalink", "bluepeak", "orbitech", "cloudway", "securenet", "medigate", "finvest",
    "greenfield", "skybridge", "paylog", "techwave", "binat", "carmel-systems", "negev-soft",
    "harel-group", "shapir", "migdal-it", "ofek-labs", "tavor-tech", "kinneret", "galil-media",
]
COMPANY_SUFFIXES = ["com", "co.il", "io", "net", "org.il"]


def latin_letters(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def latin_form(value: str, table: dict[str, str]) -> str:
    return table.get(value, value)


def email_domain(rng: random.Random) -> str:
    if rng.random() < 0.45:
        return f"{rng.choice(COMPANY_WORDS)}.{rng.choice(COMPANY_SUFFIXES)}"
    domain = rng.choice(EMAIL_DOMAINS)
    return f"{rng.choice(COMPANY_WORDS)}.{domain}" if domain == "co.il" else domain


def email_from_name(rng: random.Random, first: str | None, last: str | None) -> str:
    first_latin = latin_letters(latin_form(first, LATIN_FIRST)) if first else ""
    last_latin = latin_letters(latin_form(last, LATIN_LAST)) if last else ""
    if first_latin and last_latin:
        local = rng.choice(
            [
                f"{first_latin}.{last_latin}",
                f"{first_latin}.{last_latin}",
                f"{first_latin}{last_latin}",
                f"{first_latin}_{last_latin}",
                f"{first_latin[0]}.{last_latin}",
                f"{first_latin}.{last_latin[0]}",
                f"{last_latin}.{first_latin}",
                f"{first_latin}{last_latin}{rng.randint(1, 99)}",
                f"{first_latin}.{last_latin}{rng.randint(70, 99)}",
            ]
        )
    else:
        base = first_latin or last_latin
        local = rng.choice([base, f"{base}{rng.randint(1, 999)}", f"{base}.{rng.randint(80, 99)}", f"{base}_{rng.choice(['il', 'work', 'official'])}"])
    return f"{local}@{email_domain(rng)}"


def email(rng: random.Random) -> str:
    return email_from_name(rng, rng.choice(FIRST_NAMES)[0], rng.choice(LAST_NAMES)[0])


HEBREW_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
ENGLISH_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
HEBREW_CALENDAR_MONTHS = ["תשרי", "חשוון", "כסלו", "טבת", "שבט", "אדר", "ניסן", "אייר", "סיוון", "תמוז", "אב", "אלול"]
GEMATRIA = [
    (400, "ת"), (300, "ש"), (200, "ר"), (100, "ק"), (90, "צ"), (80, "פ"), (70, "ע"), (60, "ס"), (50, "נ"),
    (40, "מ"), (30, "ל"), (20, "כ"), (10, "י"), (9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"),
    (3, "ג"), (2, "ב"), (1, "א"),
]


def gematria(number: int) -> str:
    letters = ""
    remaining = number
    for value, letter in GEMATRIA:
        if remaining in (15, 16):
            letters += "טו" if remaining == 15 else "טז"
            break
        while remaining >= value:
            letters += letter
            remaining -= value
    if len(letters) == 1:
        return letters + "'"
    return letters[:-1] + '"' + letters[-1]


def hebrew_calendar_date(rng: random.Random) -> str:
    day = gematria(rng.randint(1, 29))
    month = rng.choice(HEBREW_CALENDAR_MONTHS)
    year = gematria(rng.randint(750, 787))
    return f"{day} ב{month} {year}"


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if year % 4 == 0 else 28
    return 30 if month in (4, 6, 9, 11) else 31


def date(rng: random.Random) -> str:
    if rng.random() < 0.04:
        return hebrew_calendar_date(rng)
    year = rng.randint(1950, 2027) if rng.random() < 0.3 else rng.randint(2015, 2027)
    month = rng.randint(1, 12)
    day = rng.randint(1, days_in_month(year, month))
    short_year = f"{year % 100:02d}"
    forms = [
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day}/{month}/{year}",
        f"{day:02d}/{month:02d}/{short_year}",
        f"{day:02d}.{month:02d}.{year}",
        f"{day}.{month}.{short_year}",
        f"{day:02d}.{month:02d}.{short_year}",
        f"{year}-{month:02d}-{day:02d}",
        f"{year}-{month:02d}-{day:02d}",
        f"{day:02d}-{month:02d}-{year}",
        f"{year}/{month:02d}/{day:02d}",
        f"{day} ב{HEBREW_MONTHS[month - 1]} {year}",
        f"{day} ב{HEBREW_MONTHS[month - 1]} {year}",
        f"{day} {HEBREW_MONTHS[month - 1]} {year}",
        f"{ENGLISH_MONTHS[month - 1]} {day}, {year}",
        f"{day} {ENGLISH_MONTHS[month - 1]} {year}",
        f"{ENGLISH_MONTHS[month - 1][:3]} {day}, {year}",
        f"{day:02d}-{ENGLISH_MONTHS[month - 1][:3]}-{year}",
        f"{year}{month:02d}{day:02d}",
    ]
    return rng.choice(forms)


def clock_time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def timestamp(rng: random.Random) -> str:
    year = rng.randint(2022, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, days_in_month(year, month))
    time_of_day = clock_time(rng)
    forms = [
        time_of_day,
        time_of_day[:5],
        f"{time_of_day}.{rng.randint(0, 999):03d}",
        f"{year}-{month:02d}-{day:02d} {time_of_day}",
        f"{year}-{month:02d}-{day:02d} {time_of_day},{rng.randint(0, 999):03d}",
        f"{day:02d}/{ENGLISH_MONTHS[month - 1][:3]}/{year}:{time_of_day} +0200",
        f"{ENGLISH_MONTHS[month - 1][:3]} {day:2d} {time_of_day}",
        f"{day:02d}.{month:02d}.{year} {time_of_day[:5]}",
        str(rng.randint(1_600_000_000, 1_800_000_000)),
    ]
    return rng.choice(forms)


def iso_datetime(rng: random.Random) -> str:
    year = rng.randint(2022, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, days_in_month(year, month))
    base = f"{year}-{month:02d}-{day:02d}T{clock_time(rng)}"
    return rng.choice(
        [
            base + "Z",
            base + "Z",
            f"{base}.{rng.randint(0, 999):03d}Z",
            f"{base}.{rng.randint(0, 999999):06d}Z",
            base + rng.choice(["+02:00", "+03:00"]),
            base,
            f"{base}.{rng.randint(0, 999):03d}+02:00",
        ]
    )


def uuid_value(rng: random.Random) -> str:
    value = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    if rng.random() < 0.1:
        return value.upper()
    if rng.random() < 0.1:
        return value.replace("-", "")
    return value


def random_hex(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(length))


def hex_hash(rng: random.Random) -> str:
    kind = rng.choice(["md5", "sha1", "sha256", "sha256", "short", "git"])
    if kind == "md5":
        value = random_hex(rng, 32)
    elif kind == "sha1":
        value = random_hex(rng, 40)
    elif kind == "sha256":
        value = random_hex(rng, 64)
        if rng.random() < 0.35:
            value = "sha256:" + value
    elif kind == "short":
        value = random_hex(rng, rng.choice([7, 8, 12, 16]))
    else:
        value = random_hex(rng, 40)[:7]
    return value.upper() if rng.random() < 0.1 else value


def alphanumeric(rng: random.Random, length: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def plain_digits(rng: random.Random, length: int) -> str:
    while True:
        value = digits(rng, length)
        if length == 9 and is_valid_israeli_id(value):
            continue
        if length in (15, 16) and is_luhn_valid(value):
            continue
        return value


LOWER_ALNUM = string.ascii_lowercase + string.digits
MIXED_ALNUM = string.ascii_letters + string.digits


def session_id(rng: random.Random) -> str:
    return rng.choice(
        [
            lambda: "sess_" + alphanumeric(rng, rng.choice([16, 24, 32]), LOWER_ALNUM),
            lambda: "sid-" + random_hex(rng, rng.choice([12, 16, 24])),
            lambda: alphanumeric(rng, rng.choice([22, 26, 32]), MIXED_ALNUM),
            lambda: random_hex(rng, rng.choice([16, 32])),
            lambda: "S" + plain_digits(rng, rng.choice([8, 9, 10, 12])),
            lambda: "SES-" + alphanumeric(rng, 10, string.ascii_uppercase + string.digits),
            lambda: plain_digits(rng, rng.choice([9, 10, 12])),
        ]
    )()


def order_ref(rng: random.Random) -> str:
    return rng.choice(
        [
            lambda: f"ORD-{digits(rng, rng.choice([5, 6, 8]))}",
            lambda: f"#{rng.choice('ABCDKRT')}-{digits(rng, 5)}",
            lambda: f"ORD-{rng.randint(2022, 2026)}-{digits(rng, 6)}",
            lambda: f"INV-{digits(rng, 5)}",
            lambda: f"REF{digits(rng, 8)}",
            lambda: f"#{digits(rng, rng.choice([4, 5, 6]))}",
            lambda: plain_digits(rng, rng.choice([5, 6, 7, 8, 9, 10, 12])),
            lambda: plain_digits(rng, rng.choice([9, 10, 16])),
            lambda: f"{rng.choice(['CS', 'TK', 'RMA', 'PO'])}-{digits(rng, rng.choice([6, 7, 9]))}",
            lambda: f"{digits(rng, 3)}-{digits(rng, 6)}",
        ]
    )()


def employee_ref(rng: random.Random) -> str:
    return rng.choice(
        [
            lambda: f"EMP-IL-{digits(rng, 5)}",
            lambda: f"E{digits(rng, 5)}",
            lambda: f"EMP{digits(rng, 6)}",
            lambda: f"IL-EMP-{digits(rng, 6)}",
            lambda: f"E-{digits(rng, 4)}",
            lambda: f"EID{digits(rng, 7)}",
            lambda: digits(rng, rng.choice([5, 6])),
            lambda: f"W{digits(rng, 7)}",
        ]
    )()


def ip_address(rng: random.Random) -> str:
    if rng.random() < 0.12:
        return ":".join(random_hex(rng, rng.choice([1, 2, 3, 4])) for _ in range(8))
    first = rng.choice([10, 192, 172, 85, 31, 62, 82, 109, 147])
    second = {192: 168, 172: rng.randint(16, 31)}.get(first, rng.randint(0, 255))
    value = f"{first}.{second}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if rng.random() < 0.12:
        value += f":{rng.choice([80, 443, 8080, 8443, 3306, 5432, 22])}"
    return value


def to_degrees_minutes(value: float, positive: str, negative: str) -> str:
    degrees = int(abs(value))
    minutes_total = (abs(value) - degrees) * 60
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60
    return f"{degrees}°{minutes:02d}'{seconds:04.1f}\"{positive if value >= 0 else negative}"


def coordinates(rng: random.Random) -> str:
    latitude = rng.uniform(29.5, 33.3)
    longitude = rng.uniform(34.2, 35.9)
    decimals = rng.choice([4, 5, 6])
    lat, lon = f"{latitude:.{decimals}f}", f"{longitude:.{decimals}f}"
    return rng.choice(
        [
            f"{lat}, {lon}",
            f"{lat},{lon}",
            f"({lat}, {lon})",
            f"lat={lat} lon={lon}",
            f"lat: {lat}, lng: {lon}",
            f"{lat}N {lon}E",
            f"{to_degrees_minutes(latitude, 'N', 'S')} {to_degrees_minutes(longitude, 'E', 'W')}",
        ]
    )


def version(rng: random.Random) -> str:
    major, minor, patch = rng.randint(0, 14), rng.randint(0, 30), rng.randint(0, 40)
    return rng.choice(
        [
            f"{major}.{minor}.{patch}",
            f"{major}.{minor}.{patch}",
            f"v{major}.{minor}.{patch}",
            f"v{major}.{minor}.{patch}-rc{rng.randint(1, 5)}",
            f"{major}.{minor}",
            f"v{major}.{minor}.{patch}-beta.{rng.randint(1, 9)}",
            f"{major}.{minor}.{patch}+build.{digits(rng, 4)}",
            f"{major}.{minor}.{patch}.{rng.randint(100, 9999)}",
        ]
    )


def serial_number(rng: random.Random) -> str:
    upper_alnum = string.ascii_uppercase + string.digits
    return rng.choice(
        [
            lambda: f"SN-{alphanumeric(rng, 8, upper_alnum)}",
            lambda: f"SN{digits(rng, rng.choice([8, 10, 12]))}",
            lambda: alphanumeric(rng, 12, upper_alnum),
            lambda: plain_digits(rng, rng.choice([9, 10, 11, 12, 13])),
            lambda: "-".join(alphanumeric(rng, 4, upper_alnum) for _ in range(3)),
            lambda: f"SN-{rng.randint(2019, 2027)}-{digits(rng, 8)}",
            lambda: luhn_number(rng, rng.choice(["35", "86"]), 15),
            lambda: f"{rng.choice('ABCDFGHJK')}{alphanumeric(rng, 2, string.digits)}{alphanumeric(rng, 7, upper_alnum)}",
        ]
    )()


def amount(rng: random.Random) -> str:
    value = rng.choice([rng.randint(5, 999), rng.randint(1000, 99999), rng.randint(100000, 9999999)])
    whole = f"{value:,}" if rng.random() < 0.8 else str(value)
    shown = f"{whole}.{rng.randint(0, 99):02d}" if rng.random() < 0.4 else whole
    return rng.choice(
        [
            f"₪{shown}",
            f"₪ {shown}",
            f"{shown} ₪",
            f'{shown} ש"ח',
            f'{shown} ש"ח',
            f"{shown} שקלים",
            f"${shown}",
            f"{shown} USD",
            f"€{shown}",
            f"NIS {shown}",
        ]
    )


COUNTRIES = load_lexicon("countries_he.txt")


def country(rng: random.Random) -> str:
    name, latin = rng.choice(COUNTRIES)
    return latin if rng.random() < 0.08 else name


# No bare "יהב" (a surname) or "ירושלים" (a city): those banks always carry the word בנק.
ISRAELI_BANK_NAMES = [
    "בנק לאומי", "בנק לאומי", "לאומי", "לאומי", "Bank Leumi", "Leumi",
    "בנק הפועלים", "בנק הפועלים", "הפועלים", "Bank Hapoalim", "Hapoalim",
    "בנק דיסקונט", "דיסקונט", "Israel Discount Bank", "Discount Bank",
    "בנק מזרחי טפחות", "מזרחי טפחות", "Mizrahi Tefahot",
    "הבנק הבינלאומי", "הבינלאומי", "First International Bank", "FIBI",
    "בנק יהב", "Bank Yahav",
    "בנק מרכנתיל", "מרכנתיל", "Mercantile Bank",
    "בנק מסד", "מסד", "Bank Massad",
    "בנק אוצר החייל", "אוצר החייל",
    "בנק ירושלים", "Bank of Jerusalem",
    "וואן זירו", "One Zero",
    "בנק הדואר",
]
FOREIGN_BANK_NAMES = [
    "Citibank", "סיטיבנק", "HSBC", "Barclays", "JPMorgan Chase", "Bank of America", "Deutsche Bank",
    "דויטשה בנק", "BNP Paribas", "UBS", "Santander", "Revolut", "Wells Fargo",
]
BANK_NAMES_AFTER_BANK_WORD = [
    name for name in dict.fromkeys(ISRAELI_BANK_NAMES) if re.search("[א-ת]", name) and "בנק" not in name
]


def bank_name(rng: random.Random) -> str:
    return rng.choice(ISRAELI_BANK_NAMES if rng.random() < 0.85 else FOREIGN_BANK_NAMES)


def bank_name_after_bank_word(rng: random.Random) -> str:
    return rng.choice(BANK_NAMES_AFTER_BANK_WORD)


REAL_COMPANY_NAMES = [
    "אלביט מערכות", "טבע", "צ'ק פוינט", "אמדוקס", "נייס", "וויקס", "מונדיי", "בזק", "סלקום", "פרטנר",
    "שופרסל", "תנובה", "שטראוס", "אסם", "אל על", "דלק", "החברה לישראל", "כיל", "הפניקס", "כלל ביטוח",
    "מגדל ביטוח", "מובילאיי", "מיקרוסופט", "גוגל", "אמזון", "אינטל",
    "Elbit Systems", "Teva", "Check Point", "Amdocs", "NICE", "Wix", "monday.com", "Mobileye", "Fiverr",
    "Microsoft", "Google", "Amazon", "Apple", "Intel", "IBM", "Oracle", "SAP", "Salesforce",
    "Cisco", "Nvidia", "Deloitte", "PwC", "KPMG",
]
# Synthetic names avoid every first name, surname and city, so none of their words is labelled elsewhere.
HEBREW_COMPANY_STEMS = [
    "נקסורה", "דאטהלינק", "בלופיק", "אורביטק", "קלאודוויי", "סקיורנט", "מדיגייט", "פינווסט", "סקייברידג'",
    "פיילוג", "טקוויב", "נגב סופט", "בינת", "אלפא פתרונות", "קו מנחה", "מעגלים", "תשתיות הצפון", "דיגיטק",
]
HEBREW_COMPANY_SUFFIXES = ["", ' בע"מ', ' בע"מ', " בע״מ", " טכנולוגיות", " מערכות", ' טכנולוגיות בע"מ', " תוכנה", " שירותים"]
LATIN_COMPANY_STEMS = [
    "Nexora", "Datalink", "Bluepeak", "Orbitech", "Cloudway", "Securenet", "Medigate", "Finvest",
    "Greenfield", "Skybridge", "Paylog", "Techwave", "Negev Soft",
]
LATIN_COMPANY_SUFFIXES = ["", " Ltd", " Ltd.", " Inc.", " Technologies", " Systems", " Group"]


def company_name(rng: random.Random) -> str:
    if rng.random() < 0.4:
        return rng.choice(REAL_COMPANY_NAMES)
    if rng.random() < 0.7:
        return rng.choice(HEBREW_COMPANY_STEMS) + rng.choice(HEBREW_COMPANY_SUFFIXES)
    return rng.choice(LATIN_COMPANY_STEMS) + rng.choice(LATIN_COMPANY_SUFFIXES)


PII_FILLERS: dict[str, Callable] = {
    "FIRST_NAME": first_name,
    "LAST_NAME": last_name,
    "ID_NUM": israeli_id,
    "PASSPORT_NUM": passport_number,
    "DRIVER_LICENSE_NUM": driver_license,
    "BANK_ACCOUNT_NUM": bank_account,
    "IBAN": iban,
    "CC_NUM": credit_card,
    "CC_PROVIDER": cc_provider,
    "PHONE_NUM": phone,
    "EMAIL": email,
    "STREET": street,
    "CITY": city,
    "POSTAL_CODE": postal_code,
    "DATE": date,
}

NEGATIVE_FILLERS: dict[str, Callable] = {
    "timestamp": timestamp,
    "iso_datetime": iso_datetime,
    "uuid": uuid_value,
    "hex_hash": hex_hash,
    "session_id": session_id,
    "order_ref": order_ref,
    "employee_ref": employee_ref,
    "ip_address": ip_address,
    "coordinates": coordinates,
    "version": version,
    "serial_number": serial_number,
    "amount": amount,
    "country": country,
    "bank_name": bank_name,
    "company_name": company_name,
}
