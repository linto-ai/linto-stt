"""Unit tests of the Whisper hallucination filter (no model needed)."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parent.parent / "linto_stt/engines/whisper/stt/processing/hallucinations.py"
_spec = importlib.util.spec_from_file_location("hallucinations", _PATH)
hallucinations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hallucinations)


def _result(text):
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4, "conf": 0.9} for i, w in enumerate(text.split())]
    return {"text": text, "language": "fr", "confidence-score": 0.9, "words": words}


def _clean(text, mode="credits"):
    return hallucinations.remove_hallucinations(_result(text), mode)["text"]


# subtitle credits, removed by default (shapes seen in production, invented sentences)
@pytest.mark.parametrize("text, expected", [
    ("on garde cette option – Sous-titrage FR 2021 Il reste un point", "on garde cette option – Il reste un point"),
    ("le planning est validé. Sous-titrage FR ? On passe à la suite", "le planning est validé. On passe à la suite"),
    ("we can meet on Tuesday, Sous-titrage Société Radio-Canada, The agenda is", "we can meet on Tuesday, The agenda is"),
    ("le budget, mais... Sous titrage Société Radio-Canada Comme prévu", "le budget, mais... Comme prévu"),
    ("Sous-titrage : Société Radio-Canada", ""),
    ("Sous-titrage ST' 501 Sous-titrage ST' 501 Sous-titrage MFP", ""),
    ("Sous-titrage ST’501 d'accord", "d'accord"),
    ("Sous-titres par Jérémy Diaz", ""),
    ("ok www.microsoft.com", "ok"),
    ("Sous-titres réalisés para la communauté d'Amara.org", ""),
    ("Subtitles by the Amara.org community", ""),
    ("Untertitel im Auftrag des ZDF, 2021", ""),
    ("Продолжение следует...", ""),
    ("Субтитры создавал DimaTorzok", ""),
    ("字幕由 Amara.org 社区提供", ""),
    ("MBC 뉴스 김성현입니다.", ""),
    ("ترجمة نانسي قنقر", ""),
])
def test_credits_are_removed(text, expected):
    assert _clean(text) == expected


VIDEO_OUTROS = [
    ("C'est bon. Merci d'avoir regardé cette vidéo !", "C'est bon."),
    ("A bientôt. J'espère que cette vidéo vous a plu. Mettez un j'aime si c'est le cas.", "A bientôt."),
    ("Thank you for watching and see you in the next video.", ""),
    ("¡Suscríbete al canal!", ""),
    ("ご視聴ありがとうございました", ""),
    ("ご 視 聴 ありがとう ございました", ""),
]


@pytest.mark.parametrize("text, expected", VIDEO_OUTROS)
def test_video_outros_removed_with_all(text, expected):
    assert _clean(text, "all") == expected


@pytest.mark.parametrize("text, _", VIDEO_OUTROS)
def test_video_outros_kept_by_default(text, _):
    # real videos say them too
    assert _clean(text) == text


@pytest.mark.parametrize("text", [
    "le sous-titrage en temps réel est disponible",
    "une interview diffusée sur Radio-Canada hier soir",
    "merci d'avoir regardé le document avant la réunion",
    "Merci beaucoup, à la prochaine.",
    "Thank you. Bye.",
    "on parle de LinTO, Twake et OpenRAG",
    "Les sous-titres par défaut sont en français",
    "the ST 500 model",
    "Спасибо.",
])
def test_real_speech_is_kept(text):
    assert _clean(text, "all") == text


def test_timestamps_of_kept_words_are_unchanged():
    res = hallucinations.remove_hallucinations(_result("bonjour Sous-titrage ST' 501 à tous"))
    assert [w["word"] for w in res["words"]] == ["bonjour", "à", "tous"]
    assert [w["start"] for w in res["words"]] == [0.0, 2.0, 2.5]


def test_no_words_no_change():
    res = {"text": "", "language": "fr", "confidence-score": 0.0, "words": []}
    assert hallucinations.remove_hallucinations(res) is res
