"""Detect the phrases Whisper hallucinates from its subtitle training data.

On silence, noise, music, or at the start of some windows even over speech,
Whisper sometimes writes a subtitle credit or a video outro instead of what is
said ("Sous-titrage Société Radio-Canada", "Sous-titrage ST' 501", "Untertitel
im Auftrag des ZDF", "Merci d'avoir regardé cette vidéo"), with a high
confidence, so no confidence or no_speech threshold catches them.

This module only detects them (it removes nothing): redecode.py uses it to find
the time ranges to decode again, and to reject a new decoding that hallucinates
again.

Two kinds of patterns, on the normalized text (lowercase, typographic
apostrophes and dashes unified, punctuation dropped except inner . @ _ : / for
URLs, one space between words):
- PHRASES: whole credits and outros, matched anywhere on word boundaries;
- SEGMENT_FRAGMENTS: truncated forms ("Sous-titrage", "Merci d'avoir regardé")
  that are also ordinary words or sentences; they count as hallucinations only
  when they make up a whole Whisper segment on their own.

Every language is matched whatever the detected language: Whisper often writes
a credit in another language than the speech. A false detection only costs a
retry in redecode.py (the original words are kept when no better decoding is
found).

Sources: LinTO production transcripts (including a meeting where these phrases
replaced 30 s of speech); the list of linto-ai/linto-stt pull request #141
(itself built from LinTO transcripts, the "Bag of Hallucinations" study,
arXiv 2501.11378, openai/whisper discussions #928 #679 #1606 #1873 #2608, and
the blocklists of Vexa, MCR, Speech-Translate and Deckle).
"""
import unicodedata

import regex as re


def _cjk(text: str) -> str:
    """Pattern for a CJK/Hangul phrase, whatever spaces the word split put in it."""
    return " ?".join(re.escape(c) for c in text.replace(" ", "").lower())


_CREDIT_WORDS = (
    r"\S*(?:sous-?titr|sous titr|soustitr|subt[ií]t|subtitl|untertitel|sottotitol|legenda|ondertitel|"
    r"napisy|undertext|transcri|translated|traduction|tłumaczenie)\S*"
)
_FR_CREDIT_TAIL = r"(?:société radio-canada|radio-canada|société|st'? ?501|mfp|fr(?: \d{4})?)"

PHRASES = [
    # Amara.org credits, any language: from the credit word up to "amara.org"
    rf"{_CREDIT_WORDS}(?: \S+){{0,10}}? \S*amara\.?org(?:[- ](?:community|communit\S*|gemeenschap|gemenskapen|gemeinschaft))?",
    r"\S*amara\.org\S*",
    _cjk("字幕由Amara.org社区提供"),
    _cjk("小編字幕由Amara.org社區提供"),
    _cjk("由Amara.org社群提供的字幕"),
    # French credits: "Sous-titrage" followed by at least one credit word
    rf"sous[- ]?titr(?:age|es?)(?: {_FR_CREDIT_TAIL})+",
    r"société radio-canada",
    r"st'? ?501",
    r"sous[- ]?titres? (?:réalisés? |faits? )?par (?:jérémy diaz|leonard m|david liou|la communauté)\S*",
    r"(?:par )?soustitreur\.com",
    # English credits
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
    # German credits
    r"(?:copyright (?:wdr|swr) \d{4} )?untertitel(?:ung)?(?: im auftrag)? des zdf(?: für funk)?(?: \d{4})?",
    r"copyright (?:wdr|swr) \d{4}",
    r"untertitel von stephanie geiges",
    r"untertitelung ärzte ohne grenzen",
    r"vertraue und glaube es hilft es heilt die göttliche kraft",
    # Spanish, Italian, Portuguese, Dutch credits
    r"(?:más información )?www\.alimmenta\.com",
    r"(?:c )?tv gelderland 2021",
    r"sottotitoli (?:e revisione )?a cura di (?:qtss|sottotitoli)",
    r"legenda adriana zanotto",
    # Russian, Ukrainian credits
    r"субтитры (?:сделал|создавал|делал|добавил|подогнал) \S+",
    r"\S*dimatorzok\S*",
    r"редактор субтитров \S+(?: корректор \S+)?",
    r"спасибо за субтитры \S+ \S+",
    r"субтитрувальниця оля шор",
    # Chinese, Korean, Arabic, other credits
    _cjk("中文字幕志愿者杨茜茜"),
    _cjk("字幕志愿者李宗盛"),
    _cjk("자막 제공"),
    r"mbc 뉴스 \S+입니다",
    r"ترجمة نانسي قنقر",
    r"napisy by jacek makarewicz",
    r"altyazı m\.k\.?",
    r"textning\.nu",
    r"(?:norsk )?teksting av nicolai winther|tekstet av nicolai winther|undertekster av ai-media",
    r"titulky vytvořil johnyx",
    # Channel URLs written on silence
    r"www\.(?:microsoft\.com|americasrollercoasters\.tatepublishing\.net|feyyaz\.tv|mooji\.org|multi-moto\.eu|subsedit\.dk|sdimedia\.com|legendavie\.ro)",
    # Video outros (French, English, German, Spanish, Italian, Portuguese, Dutch,
    # Russian, Ukrainian, Arabic, Japanese, Chinese, Korean, others)
    r"merci (?:beaucoup )?d'avoir regardé (?:cette vidéo|la vidéo|cet épisode)",
    r"merci à (?:mes|tous mes|nos) tipeurs(?: tipeee)?(?: et (?:zephyr|souscripteurs|abonnés))?",
    r"j'espère que (?:vous avez apprécié (?:la|cette) vidéo|(?:la|cette) vidéo vous a plu)",
    r"n'oubliez pas de (?:vous abonner|liker|mettre un like)(?: à la chaîne)?",
    r"abonnez-vous (?:à (?:la|ma|notre) chaîne|et activez la cloche)",
    r"mettez un (?:j'aime|pouce bleu|like) si c'est le cas",
    r"je vous remercie de vous abonner",
    r"bonjour à tous et bienvenue sur ma chaîne",
    r"(?:thank you|thanks) (?:so much |very much )?for watching(?: and (?:i'll )?see you (?:next time|in the next (?:video|one)))?",
    r"(?:please )?like and subscribe",
    r"please subscribe(?: to (?:my|our|the) channel)?",
    r"subscribe to (?:my|our|the) channel",
    r"(?:hello everyone|hi guys) welcome (?:back )?to my channel",
    r"(?:vielen )?dank(?:e)? f(?:ü|u)rs (?:zuschauen|ansehen)",
    r"vielen dank für das zuschauen",
    r"abonniere meinen kanal",
    r"hallo zusammen und willkommen auf meinem kanal",
    r"(?:muchas )?gracias por ver (?:el|este) v[ií]deo",
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
    r"спасибо за просмотр",
    r"подписывайтесь на (?:мой|наш) канал",
    r"ставьте лайки и подписывайтесь",
    r"дякую за перегляд",
    r"підписуйтесь на наш канал",
    r"اشتركوا في القناة",
    r"شكرا (?:على|ل)ل?مشاهدة",
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

# Truncated credits and outros: hallucinations only when they are a whole
# segment ("le sous-titrage en temps réel" or "merci d'avoir regardé le
# document" are real speech).
SEGMENT_FRAGMENTS = [
    r"sous[- ]?titr(?:age|es?)",
    r"merci (?:beaucoup )?d'avoir regardé",
    r"(?:thank you|thanks) for watching",
    r"untertitel(?:ung)?",
    r"subtitles?",
    r"продолжение следует",
]

_PHRASES = re.compile("|".join(rf"(?<!\S)(?:{p})(?!\S)" for p in PHRASES))
_FRAGMENT = re.compile("|".join(rf"(?:{p})" for p in SEGMENT_FRAGMENTS))

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "‐": "-", "‑": "-", "–": "-", "—": "-"})
_STRIP = re.compile(r"[^\p{L}\p{N}\p{M}'\-.@_:/]+")


def normalize(word: str) -> str:
    word = unicodedata.normalize("NFC", word).translate(_APOSTROPHES).lower()
    word = _STRIP.sub("", word)
    return word.strip(".-:")


def find_hallucinated_words(words: list, segments: list = None) -> set:
    """Indexes of the words that belong to a known hallucinated phrase.

    `words` is a list of word strings (as written, possibly with punctuation);
    `segments` an optional list of (first, last) word indexes of each Whisper
    segment, to detect the fragments that make up a whole segment."""
    normalized = [normalize(w) for w in words]
    tokens, spans, pos = [], [], 0
    for i, t in enumerate(normalized):
        if not t:
            continue
        spans.append((pos, pos + len(t), i))
        tokens.append(t)
        pos += len(t) + 1
    text = " ".join(tokens)
    covered = set()
    # overlapped: a phrase may start inside another match (two phrases in a row)
    for m in _PHRASES.finditer(text, overlapped=True):
        covered.update(i for start, end, i in spans if start >= m.start() and end <= m.end())
    for first, last in segments or []:
        segment = " ".join(t for t in normalized[first:last + 1] if t)
        if segment and _FRAGMENT.fullmatch(segment):
            covered.update(range(first, last + 1))
    # a punctuation-only word right after a detected phrase goes with it ("FR ?", "vidéo !")
    for i, t in enumerate(normalized):
        if i - 1 in covered and not t:
            covered.add(i)
    return covered
