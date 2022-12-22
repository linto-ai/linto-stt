import math
import re
# import string
import unicodedata
from num2words import num2words

from stt import logger
from .utils import flatten

# string.punctuation, plus Whisper specific "«»¿", minus apostrophe "'", dash "-", and dot "." (which will be processed as special)
_punctuations = '!"#$%&()*+,/:;<=>?@[\\]^_`{|}~«»¿'


def remove_punctuation(text: str) -> str:
    text = text.translate(str.maketrans("", "", _punctuations))
    # We don't remove dots inside words (e.g. "ab@gmail.com")
    text = re.sub(r"\.(\s)", r"\1", text+" ").strip()
    return collapse_whitespace(text)


_whitespace_re = re.compile(r'[^\S\r\n]+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()


def transliterate(c):
    # Transliterates a character to its closest ASCII equivalent.
    # Example: transliterate("à ß œ ﬂ") = "a ss oe fl"
    c = re.sub("œ", "oe", c)
    c = re.sub("æ", "ae", c)
    c = re.sub("Œ", "OE", c)
    c = re.sub("Æ", "AE", c)
    c = re.sub("ß", "ss", c)
    return unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")


def remove_emoji(text):
    # Remove emojis
    return re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", "", text)


def normalize_text(text: str, lang: str) -> str:
    """ Transform digits into characters... """

    # Reorder currencies (1,20€ -> 1 € 20)
    coma = "," if lang in ["fr"] else "\."
    for c in _currencies:
        if c in text:
            text = re.sub(r"\b(\d+)" + coma + r"(\d+)\s*" +
                          c, r"\1 " + c + r" \2", text)

    # Roman digits
    if re.search(r"[IVX]", text):
        if lang == "en":
            digits = re.findall(
                r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(º|st|nd|rd|th)?\b", text)
            digits = ["".join(d) for d in digits]
        elif lang == "fr":
            digits = re.findall(
                r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(º|ème|eme|e|er|ère)?\b", text)
            digits = ["".join(d) for d in digits]
        else:
            digits = []
        if digits:
            digits = sorted(list(set(digits)), reverse=True,
                            key=lambda x: (len(x), x))
            for s in digits:
                filtered = re.sub("[a-z]", "", s)
                ordinal = filtered != s
                digit = roman_to_decimal(filtered)
                v = undigit(str(digit), lang=lang,
                            to="ordinal" if ordinal else "cardinal")
                text = re.sub(r"\b" + s + r"\b", v, text)

    # Ordinal digits
    if lang == "en":
        digits = re.findall(
            r"\b\d*1(?:st)|\d*2(?:nd)|\d*3(?:rd)|\d+(?:º|th)\b", text)
    elif lang == "fr":
        digits = re.findall(
            r"\b1(?:ère|ere|er|re|r)|2(?:nd|nde)|\d+(?:º|ème|eme|e)\b", text)
    else:
        logger.warn(
            f"Language {lang} not supported for normalization. Some words might be mis-localized.")
        digits = []
    if digits:
        digits = sorted(list(set(digits)), reverse=True,
                        key=lambda x: (len(x), x))
        for digit in digits:
            word = undigit(re.findall(r"\d+", digit)
                           [0], to="ordinal", lang=lang)
            text = re.sub(r'\b'+str(digit)+r'\b', word, text)

    # Cardinal digits
    digits = re.findall(
        r"(?:\-?\b[\d/]*\d+(?: \d\d\d)+\b)|(?:\-?\d[/\d]*)", text)
    digits = list(map(lambda s: s.strip(r"[/ ]"), digits))
    digits = list(set(digits))
    digits = digits + flatten([c.split() for c in digits if " " in c])
    digits = digits + flatten([c.split("/") for c in digits if "/" in c])
    digits = sorted(digits, reverse=True, key=lambda x: (len(x), x))
    for digit in digits:
        digitf = re.sub("/+", "/", digit)
        if not digitf:
            continue
        numslash = len(re.findall("/", digitf))
        if numslash == 0:
            word = undigit(digitf, lang=lang)
        elif numslash == 1:  # Fraction or date
            i = digitf.index("/")
            is_date = False
            if len(digitf[i+1:]) == 2:
                try:
                    first = int(digitf[:i])
                    second = int(digitf[i+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13
                except:
                    pass
            if is_date:
                first = digitf[:i].lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (
                    lang != "fr" and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang,
                                to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month[second]
            else:
                first = undigit(digitf[:i], lang=lang)
                second = undigit(digitf[i+1:], to="denominator", lang=lang)
                if float(digitf[:i]) > 2. and second[-1] != "s":
                    second += "s"
            word = first + " " + second
        elif numslash == 2:  # Maybe a date
            i1 = digitf.index("/")
            i2 = digitf.index("/", i1+1)
            is_date = False
            if len(digitf[i1+1:i2]) == 2 and len(digitf[i2+1:]) == 4:
                try:
                    first = int(digitf[:i1])
                    second = int(digitf[i1+1:i2])
                    third = int(digitf[i2+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13 and third > 1000
                except:
                    pass
            third = undigit(digitf[i2+1:], lang=lang)
            if is_date:
                first = digitf[:i].lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (
                    lang != "fr" and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang,
                                to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get(lang, {}).get(
                    int(digitf[i1+1:i2]), digitf[i1+1:i2])
                word = " ".join([first, second, third])
            else:
                word = " / ".join([undigit(s, lang=lang)
                                  for s in digitf.split('/')])
        else:
            word = " / ".join([undigit(s, lang=lang)
                              for s in digitf.split('/')])
        if " " in digit:
            text = re.sub(r'\b'+str(digit)+r'\b', " "+word+" ", text)
        else:
            text = re.sub(str(digit), " "+word+" ", text)

    # Symbols (currencies, percent...)
    symbol_table = _symbol_to_word.get(lang, {})
    for k, v in symbol_table.items():
        text = re.sub(k, " "+v+" ", text)

    return collapse_whitespace(text)


def undigit(str, lang, to="cardinal"):
    str = re.sub(" ", "", str)
    if to == "denominator":
        if lang == "fr":
            if str == "2":
                return "demi"
            if str == "3":
                return "tiers"
            if str == "4":
                return "quart"
        elif lang == "en":
            if str == "2":
                return "half"
            if str == "4":
                return "quarter"
        elif lang == "es":
            if str == "2":
                return "mitad"
            if str == "3":
                return "tercio"
        to = "ordinal"
    if str.startswith("0") and to == "cardinal":
        numZeros = len(re.findall(r"0+", str)[0])
        if numZeros < len(str):
            return numZeros * (robust_num2words(0, lang=lang)+" ") + robust_num2words(float(str), lang=lang, to=to)
    return robust_num2words(float(str), lang=lang, to=to)


def robust_num2words(x, lang, to="cardinal", orig=""):
    """
    Bugfix for num2words
    """
    try:
        res = num2words(x, lang=lang, to=to)
        if lang == "fr" and to == "ordinal":
            res = res.replace("vingtsième", "vingtième")
        return res
    except OverflowError:
        if x == math.inf:  # !
            return " ".join(robust_num2words(xi, lang=lang, to=to) for xi in orig)
        if x == -math.inf:  # !
            return "moins " + robust_num2words(-x, lang=lang, to=to, orig=orig.replace("-", ""))
        # TODO: print a warning
        return robust_num2words(x//10, lang=lang, to=to)


def roman_to_decimal(str):
    def value(r):
        if (r == 'I'):
            return 1
        if (r == 'V'):
            return 5
        if (r == 'X'):
            return 10
        if (r == 'L'):
            return 50
        if (r == 'C'):
            return 100
        if (r == 'D'):
            return 500
        if (r == 'M'):
            return 1000
        return -1

    res = 0
    i = 0
    while (i < len(str)):
        s1 = value(str[i])
        if (i + 1 < len(str)):
            s2 = value(str[i + 1])
            if (s1 >= s2):
                # Value of current symbol is greater or equal to the next symbol
                res = res + s1
                i = i + 1
            else:
                # Value of current symbol is greater or equal to the next symbol
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res


_int_to_month = {
    "fr": {
        1: "janvier",
        2: "février",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "août",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "décembre",
    },
    "en": {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }
}

_currencies = ["€", "$", "£", "¥"]

_symbol_to_word = {
    "fr": {
        "%": "pour cents",
        "÷": "divisé par",
        "\*": "fois",  # ?
        "×": "fois",
        "±": "plus ou moins",
        "\+": "plus",
        "&": "et",
        "@": "arobase",
        "m²": "mètres carrés",
        "m³": "mètres cubes",
        "²": "au carré",
        "³": "au cube",
        "¼": "un quart",
        "½": "un demi",
        "¾": "trois quarts",
        "§": "section",
        "°C": "degrés Celsius",
        "°F": "degrés Fahrenheit",
        "°K": "kelvins",
        "°": "degrés",
        "€": "euros",
        "¢": "cents",
        "\$": "dollars",
        "£": "livres",
        "¥": "yens",
        # Below: not in Whisper tokens
        # "₩": "wons",
        # "₽": "roubles",
        # "₹": "roupies",
        # "₺": "liras",
        # "₪": "shekels",
        # "₴": "hryvnias",
        # "₮": "tugriks",
        # "℃": "degrés Celsius",
        # "℉": "degrés Fahrenheit",
        # "Ω": "ohms",
        # "Ω": "ohms",
        # "K": "kelvins",
        # "ℓ": "litres",
    },
    "en": {
        "%": "percent",
        "÷": "divided by",
        "\*": "times",  # ?
        "×": "times",
        "±": "plus or minus",
        "\+": "plus",
        "&": "and",
        "@": "at",
        "m²": "square meters",
        "m³": "cubic meters",
        "²": "squared",
        "³": "cubed",
        "¼": "one quarter",
        "½": "one half",
        "¾": "three quarters",
        "§": "section",
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
        "°K": "kelvins",
        "°": "degrees",
        "€": "euros",
        "¢": "cents",
        "\$": "dollars",
        "£": "pounds",
        "¥": "yens",
    }
}
