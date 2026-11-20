"""The hotwords format, common to all engines.

A request's `hotwords` (or the HOTWORDS default) is one string: terms separated
by commas, each term possibly made of several words ("Twake, Jean-Luc Machin").
Without a comma, the whole string is one term ("chambre double").
linto-transcription-service sends a list of terms joined with ", ".
"""


def parse_hotwords(hotwords):
    """The list of terms of a hotwords string (empty for None or "")."""
    if not hotwords:
        return []
    return [term.strip() for term in hotwords.split(",") if term.strip()]
