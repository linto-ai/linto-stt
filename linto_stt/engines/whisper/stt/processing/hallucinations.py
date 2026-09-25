"""Remove the fixed phrases Whisper hallucinates from its subtitle training data.

On silence, noise, music or right after a speech clip, Whisper sometimes writes
subtitle credits or video boilerplate ("Sous-titrage ST' 501", "Untertitel im
Auftrag des ZDF", "Merci d'avoir regardé cette vidéo"...) with high confidence,
so no confidence or no_speech threshold catches them. They are matched here as
whole phrases and their words are dropped.

Two lists. Subtitle credits and URLs (removed by default) are never real speech.
Video outros ("Thanks for watching", "Abonnez-vous") are also said in real
videos, so they are removed only with HALLUCINATION_FILTER=all. Short polite
formulas ("Merci.", "Thank you.", "À la prochaine") are never listed.
HALLUCINATION_FILTER=false disables the filter.
"""
import os
import unicodedata

import regex as re

_MODE = os.environ.get("HALLUCINATION_FILTER", "credits").strip().lower()
HALLUCINATION_FILTER = {"true": "credits", "1": "credits", "credits": "credits", "all": "all"}.get(_MODE, False)

# Patterns match the normalized text: lowercase, typographic apostrophes and
# dashes unified, other punctuation removed (inner . @ _ : / kept for URLs),
# one space between words. Sources: LinTO production transcripts, the "Bag of
# Hallucinations" study (arXiv 2501.11378), openai/whisper discussions #928 #679
# #1606 #1873 #2608, and the blocklists of Vexa, MCR, Speech-Translate and Deckle.
# Every language is matched whatever the detected language, since Whisper often
# writes another language's credit.


def _cjk(text: str) -> str:
    """Pattern for a CJK/Hangul phrase, whatever spaces the word split put in it."""
    return " ?".join(re.escape(c) for c in text.replace(" ", "").lower())


_CREDIT_WORDS = (
    r"\S*(?:sous-?titr|sous titr|soustitr|subt[ií]t|subtitl|untertitel|sottotitol|legenda|ondertitel|"
    r"napisy|undertext|transcri|translated|traduction|tłumaczenie|zdjęcia|phụ|cliquez-vous)\S*"
)

CREDIT_PATTERNS = [
    # Amara.org credits, any language: from the credit word up to "amara.org"
    rf"{_CREDIT_WORDS}(?: \S+){{0,10}}? \S*amara\.?org(?:[- ](?:community|communit\S*|gemeenschap|gemenskapen|gemeinschaft))?",
    r"\S*amara\.org\S*",
    _cjk("字幕由Amara.org社区提供"),
    _cjk("小編字幕由Amara.org社區提供"),
    _cjk("由Amara.org社群提供的字幕"),
    # French
    r"sous[- ]?titr(?:age|es?)(?: (?:société radio-canada|st'? ?501|mfp|fr(?: \d{4})?))+",
    r"société radio-canada",
    r"st'? ?501",
    r"sous[- ]?titres par (?:jérémy diaz|leonard m|david liou)",
    r"(?:par )?soustitreur\.com",
    # English
    r"subtitles by steamteamextra",
    r"transcribed by https?://otter\.ai",
    r"transcription by castingwords",
    r"closed captioning (?:provided )?by (?:the imperial news network|muhsen|krisbrandhagen\.com)",
    r"captions by (?:nicosubs|gettranscribed\.com)",
    r"captioned by cotter captioning services",
    r"satsang with mooji",
    r"(?:for more information )?visit www\.fema\.gov",
    r"jack link's jerky feed your wild side",
    r"blank_?audio",
    # German
    r"(?:copyright (?:wdr|swr) \d{4} )?untertitel(?:ung)?(?: im auftrag)? des zdf(?: für funk)?(?: \d{4})?",
    r"copyright (?:wdr|swr) \d{4}",
    r"untertitel von stephanie geiges",
    r"untertitelung ärzte ohne grenzen",
    r"vertraue und glaube es hilft es heilt die göttliche kraft",
    # Spanish, Italian, Portuguese, Dutch
    r"(?:más información )?www\.alimmenta\.com",
    r"sottotitoli (?:e revisione )?a cura di (?:qtss|sottotitoli)",
    r"legenda adriana zanotto",
    r"(?:c )?tv gelderland 2021",
    # Russian, Ukrainian
    r"продолжение следует",
    r"субтитры (?:сделал|создавал|делал|добавил|подогнал) \S+",
    r"\S*dimatorzok\S*",
    r"редактор субтитров \S+(?: корректор \S+)?",
    r"спасибо за субтитры \S+ \S+",
    r"субтитрувальниця оля шор",
    # Chinese, Korean, Arabic
    _cjk("中文字幕志愿者杨茜茜"),
    _cjk("字幕志愿者李宗盛"),
    _cjk("자막 제공"),
    r"mbc 뉴스 \S+입니다",
    r"ترجمة نانسي قنقر",
    # Other languages
    r"napisy by jacek makarewicz",
    r"altyazı m\.k\.?",
    r"textning\.nu",
    r"(?:norsk )?teksting av nicolai winther|tekstet av nicolai winther|undertekster av ai-media",
    r"titulky vytvořil johnyx",
    # Channel URLs written on silence
    r"www\.(?:microsoft\.com|americasrollercoasters\.tatepublishing\.net|feyyaz\.tv|mooji\.org|multi-moto\.eu|subsedit\.dk|sdimedia\.com|legendavie\.ro)",
]

# Video outros and channel boilerplate: Whisper writes them on silence too, but
# real videos uploaded for transcription also say them.
VIDEO_PATTERNS = [
    # French
    r"merci (?:beaucoup )?d'avoir regardé (?:cette vidéo|la vidéo|cet épisode)(?: merci)?(?: à bientôt)?",
    r"j'espère que (?:vous avez apprécié (?:la|cette) vidéo|(?:la|cette) vidéo vous a plu)",
    r"mettez un (?:j'aime|pouce bleu|like) si c'est le cas",
    r"n'oubliez pas de (?:vous abonner|liker|mettre un like)(?: à la chaîne)?",
    r"je vous remercie de vous abonner",
    r"abonnez-vous (?:à (?:la|ma|notre) chaîne|et activez la cloche)",
    r"bonjour à tous et bienvenue sur ma chaîne",
    r"merci à mes tipeurs(?: tipeee)?(?: et zephyr)?",
    # English
    r"(?:thank you|thanks) (?:so much |very much )?for watching(?: and (?:i'll )?see you (?:next time|in the next (?:video|one)))?",
    r"(?:please )?like and subscribe",
    r"please subscribe(?: to (?:my|our|the) channel)?",
    r"subscribe to (?:my|our|the) channel",
    r"(?:hello everyone|hi guys) welcome (?:back )?to my channel",
    # German, Spanish, Italian, Portuguese, Dutch
    r"(?:vielen )?dank(?:e)? f(?:ü|u)rs (?:zuschauen|ansehen)",
    r"vielen dank für das zuschauen",
    r"abonniere meinen kanal",
    r"hallo zusammen und willkommen auf meinem kanal",
    r"(?:muchas )?gracias por ver(?: (?:el|este) v[ií]deo)?",
    r"suscr[ií]bete al canal(?: y dale a like)?",
    r"no olvides suscribirte",
    r"hola a todos bienvenidos a mi canal",
    r"grazie per (?:la visione|aver guardato il video)",
    r"iscriviti al canale",
    r"non dimenticare di iscriverti",
    r"ciao a tutti e benvenuti sul mio canale",
    r"obrigad[oa] por assistir",
    r"(?:se )?inscreva(?:-se)? no canal",
    r"bedankt voor het kijken",
    # Russian, Ukrainian, Arabic
    r"спасибо за просмотр",
    r"подписывайтесь на (?:мой|наш) канал",
    r"ставьте лайки и подписывайтесь",
    r"дякую за перегляд",
    r"підписуйтесь на наш канал",
    r"اشتركوا في القناة",
    r"شكرا (?:على|ل)ل?مشاهدة",
    # Japanese, Chinese, Korean
    *[_cjk(t) for t in (
        "ご視聴ありがとうございました", "ご視聴ありがとうございます", "ご視聴をありがとうございます",
        "最後までご視聴いただきありがとうございました", "チャンネル登録をお願いします",
        "チャンネル登録をよろしくお願いします", "チャンネル登録よろしくお願いします",
        "高評価とチャンネル登録をお願いします", "次の動画でお会いしましょう",
        "また次回の動画でお会いしましょう", "見てくれてありがとう",
        "请不吝点赞订阅转发打赏支持明镜与点点栏目", "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
        "谢谢观看下集再见", "謝謝觀看下次見", "谢谢观看", "感谢观看", "本期視頻就拍到這裡了感謝觀看", "记得订阅",
        "시청해주셔서 감사합니다", "오늘도 영상 봐주셔서 감사합니다", "구독과 좋아요 부탁드려요",
        "구독해주세요", "채널 구독 부탁드립니다", "구독과 좋아요는 사랑입니다", "다음 영상에서 만나요",
        "이 영상은 유료광고를 포함하고 있습니다",
    )],
    # Other languages
    r"dziękuję za oglądanie",
    r"abone olmayı unutmayın",
    r"izlediğiniz için teşekkür ederim",
    r"bir sonraki videoda görüşmek üzere",
    r"ευχαριστώ που παρακολουθήσατε",
    r"vă mulțumim pentru vizionare",
    r"terima kasih telah menonton",
    r"kiitos kun katsoit(?: videon)?",
    r"köszönöm hogy megnézted",
]


def _compile(patterns):
    # a phrase starts and ends on word boundaries of the normalized text
    return re.compile("|".join(rf"(?<!\S)(?:{p})(?!\S)" for p in patterns))


_REGEX = {"credits": _compile(CREDIT_PATTERNS), "all": _compile(CREDIT_PATTERNS + VIDEO_PATTERNS)}

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "‐": "-", "‑": "-", "–": "-", "—": "-"})
_STRIP = re.compile(r"[^\p{L}\p{N}\p{M}'\-.@_:/]+")


def _normalize(word: str) -> str:
    word = unicodedata.normalize("NFC", word).translate(_APOSTROPHES).lower()
    word = _STRIP.sub("", word)
    return word.strip(".-:")


def find_hallucinated_words(words: list, mode: str = "credits") -> set:
    """Indexes of the words covered by a known hallucination phrase."""
    tokens, spans, pos = [], [], 0
    for i, w in enumerate(words):
        t = _normalize(w)
        if not t:
            continue
        spans.append((pos, pos + len(t), i))
        tokens.append(t)
        pos += len(t) + 1
    text = " ".join(tokens)
    covered = set()
    for m in _REGEX[mode].finditer(text):
        for start, end, i in spans:
            if start >= m.start() and end <= m.end():
                covered.add(i)
    # a punctuation-only word right after a removed phrase goes with it ("FR ?", "vidéo !")
    for i, w in enumerate(words):
        if i - 1 in covered and not _normalize(w):
            covered.add(i)
    return covered


def remove_hallucinations(result: dict, mode: str = "credits") -> dict:
    """Drop the words of known hallucinated phrases from a decode() result."""
    words = result.get("words") or []
    if not words:
        return result
    covered = find_hallucinated_words([w["word"] for w in words], mode)
    if not covered:
        return result
    kept = [w for i, w in enumerate(words) if i not in covered]
    result = dict(result)
    result["words"] = kept
    result["text"] = " ".join(w["word"] for w in kept).strip()
    return result
