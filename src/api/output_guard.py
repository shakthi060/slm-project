from __future__ import annotations

import re
from collections import Counter

LIST_ITEM_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*)$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_PATTERN = re.compile(r"[a-z0-9']+")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s']+")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")

SAFE_FALLBACK = (
    "I could not produce a reliable answer from the ShopEase policy. "
    "Please rephrase your question or contact ShopEase support for help."
)

SAFE_OUT_OF_SCOPE = (
    "I can only help with ShopEase customer support. I can assist with orders, "
    "delivery, returns, refunds, payments, products, warranty, or account questions."
)


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = LIST_PREFIX_PATTERN.sub("", lowered)
    lowered = NON_WORD_PATTERN.sub(" ", lowered)
    return " ".join(lowered.split())


def detect_repetition(text: str) -> list[str]:
    reasons: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_items = []
    for line in lines:
        match = LIST_ITEM_PATTERN.match(line)
        if match:
            item = _normalize(match.group(1))
            if len(item) >= 10:
                list_items.append(item)

    if len(list_items) >= 6:
        counts = Counter(list_items)
        if any(count >= 3 for count in counts.values()):
            reasons.append("repeated_list_items")
        if len(set(list_items)) / len(list_items) < 0.5:
            reasons.append("low_list_item_diversity")

    sentences = []
    for sentence in SENTENCE_SPLIT_PATTERN.split(text):
        normalized = _normalize(sentence)
        if len(normalized) >= 15:
            sentences.append(normalized)
    if len(sentences) >= 5:
        counts = Counter(sentences)
        if any(count >= 3 for count in counts.values()):
            reasons.append("repeated_sentences")
        if len(set(sentences)) / len(sentences) < 0.5:
            reasons.append("low_sentence_diversity")

    words = WORD_PATTERN.findall(text.lower())
    if len(words) >= 30:
        if len(set(words)) / len(words) < 0.25:
            reasons.append("low_word_diversity")
        trigrams = Counter(
            tuple(words[index : index + 3])
            for index in range(len(words) - 2)
        )
        if any(count >= 4 for count in trigrams.values()):
            reasons.append("repeated_phrases")

    return reasons


def is_malformed(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True
    if len(stripped) > 4000:
        return True
    return False
