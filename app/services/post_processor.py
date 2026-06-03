import re


DISFLUENCIES = re.compile(
    r"\b(?:um|uh|er|ah|hmm|mm[-\s]?hmm|uh[-\s]?huh|you know|i mean|sort of|kind of|you see|like)\b\.?,?\s*",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
MULTIPLIERS = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
    "fortieth": "40th", "fiftieth": "50th", "sixtieth": "60th",
    "seventieth": "70th", "eightieth": "80th", "ninetieth": "90th",
}
MONTH_NAMES = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}
CURRENCY_MAP = {
    "dollar": "$", "dollars": "$", "cent": "\u00a2", "cents": "\u00a2",
    "euro": "\u20ac", "euros": "\u20ac", "pound": "\u00a3", "pounds": "\u00a3",
}
PROFANITY_LIST = [
    "fuck", "shit", "damn", "bitch", "ass", "bastard", "crap", "dick",
    "piss", "slut", "whore", "bullshit", "asshole",
]


class PostProcessor:
    def __init__(self, config):
        self.config = config
        self._profanity_re = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in PROFANITY_LIST) + r")\b",
            re.IGNORECASE,
        )

    def process(self, text):
        text = self.remove_disfluencies(text)
        text = self.normalize_numbers(text)
        text = self.normalize_currency(text)
        text = self.normalize_ordinals(text)
        text = self.restore_capitalization(text)
        text = self.filter_profanity(text)
        return text.strip()

    def remove_disfluencies(self, text):
        return DISFLUENCIES.sub("", text).strip()

    def normalize_numbers(self, text):
        def replace_match(m):
            words = m.group(1).split()
            val = self._parse_number_words(words)
            return str(val) if val is not None else m.group(0)

        text = re.sub(
            r"\b((?:" + "|".join(NUMBER_WORDS.keys()) + r"|"
            + "|".join(MULTIPLIERS.keys()) + r"\b\s*)+)\b",
            replace_match,
            text,
            flags=re.IGNORECASE,
        )
        return text

    def _parse_number_words(self, words):
        total = 0
        current = 0
        for w in words:
            w_low = w.lower()
            if w_low in NUMBER_WORDS:
                current += NUMBER_WORDS[w_low]
            elif w_low in MULTIPLIERS:
                mult = MULTIPLIERS[w_low]
                if current == 0:
                    current = 1
                current *= mult
                if mult >= 1000:
                    total += current
                    current = 0
            else:
                return None
        return total + current

    def normalize_currency(self, text):
        def replace_currency(m):
            amount_text = m.group(1).strip()
            currency_word = m.group(2).lower()
            sym = CURRENCY_MAP.get(currency_word, currency_word)
            amount = self._parse_number_words(amount_text.split())
            if amount is not None:
                return f"{sym}{amount:,}" if sym in ("$", "\u00a3", "\u20ac") else f"{amount}{sym}"
            return m.group(0)

        text = re.sub(
            r"\b((?:" + "|".join(NUMBER_WORDS.keys()) + r"\s*)+)\s+("
            + "|".join(CURRENCY_MAP.keys()) + r")\b",
            replace_currency,
            text,
            flags=re.IGNORECASE,
        )
        return text

    def normalize_ordinals(self, text):
        def replace_ordinal(m):
            word = m.group(1).lower()
            return ORDINAL_WORDS.get(word, m.group(0))

        pattern = r"\b(" + "|".join(ORDINAL_WORDS.keys()) + r")\b"
        return re.sub(pattern, replace_ordinal, text, flags=re.IGNORECASE)

    def restore_capitalization(self, text):
        if not text:
            return text
        return text[0].upper() + text[1:]

    def filter_profanity(self, text):
        return self._profanity_re.sub("****", text)
