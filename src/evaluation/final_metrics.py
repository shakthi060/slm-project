from __future__ import annotations

import re
from collections import Counter

from src.evaluation.baseline import COVERAGE_KEYWORDS, classify
from src.evaluation.transformer_baseline import parse_category


ORDER_ID_PATTERN = re.compile(r"\bse-\d{8}\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
)
CURRENCY_PATTERN = re.compile(r"(?:INR|Rs\.?|₹)\s?([\d,]+)", re.IGNORECASE)
DATE_PATTERNS = [
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

ALLOWED_EMAILS = {"support@shopease.example"}
GENERIC_ALLOWED_ORDER_IDS = {"SE-48219376", "SE-12345678"}

NEGATION_TOKENS = (
    "not",
    "cannot",
    "can't",
    "never",
    "no ",
    "unable",
    "don't",
    "does not",
    "do not",
    "isn't",
    "aren't",
    "won't",
    "without",
    "should not",
    "must not",
    "avoid",
)

CONDITIONAL_PREFIXES = ("if ", "once ", "when ", "after ")
ECHO_PREFIXES = (
    "i understand",
    "you mentioned",
    "you said",
    "you reported",
    "since ",
    "because ",
    "as your ",
)
ECHO_PHRASES = (
    "you mentioned",
    "you said",
    "you reported",
    "as you noted",
    "i understand",
)

FLAG_PATTERNS = {
    "coupon_stacking_claim": re.compile(
        r"\b(?:combine|stack|merge)\b.{0,40}\bcoupons?\b|"
        r"\bcoupons?\b.{0,40}\b(?:combine|stack|merge)\b",
        re.IGNORECASE,
    ),
    "instant_or_cash_refund_claim": re.compile(
        r"\b(?:instant|cash)\s+refunds?\b", re.IGNORECASE
    ),
    "same_day_or_international_claim": re.compile(
        r"\b(?:same[- ]day|international)\s+(?:delivery|shipping)\b",
        re.IGNORECASE,
    ),
    "phone_support_claim": re.compile(
        r"\b(?:phone support|call us|call our|phone number)\b", re.IGNORECASE
    ),
    "price_match_claim": re.compile(
        r"\bprice[- ]match(?:ing| guarantee)?\b", re.IGNORECASE
    ),
    "cancel_after_dispatch_claim": re.compile(
        r"\bcancel\b.{0,60}\b(?:after dispatch|after it has shipped|"
        r"after shipping|once dispatched)\b|"
        r"\b(?:after dispatch|after it has shipped|after shipping|once dispatched)\b"
        r".{0,60}\bcancel\b",
        re.IGNORECASE,
    ),
    "account_merge_claim": re.compile(
        r"\bmerge\b.{0,40}\baccounts?\b|\baccounts?\b.{0,40}\bmerge\b",
        re.IGNORECASE,
    ),
    "requests_secret_claim": re.compile(
        r"\b(?:share|provide|send|tell|give)\b.{0,40}\b(?:otp|pin|password)\b",
        re.IGNORECASE,
    ),
}

ORDER_STATUS_SPECIFIC = re.compile(
    r"\b(?:your|the)\s+order\s+(?:was|is)\s+(?:delivered|shipped)\b|"
    r"\b(?:delivery date|out for delivery|will arrive on|arrived on|shipped on)\b",
    re.IGNORECASE,
)
RETURN_WINDOW_MISMATCH = re.compile(
    r"\b(?:24|72)\s*(?:-|to)?\s*(?:24|72)?\s*hours?\b", re.IGNORECASE
)
WARRANTY_DAMAGE_CLAIM = re.compile(
    r"\b(?:physical|liquid|accidental)\s+damage\b.{0,60}"
    r"\b(?:covered|cover|included)\b|"
    r"\b(?:covered|cover|included)\b.{0,60}"
    r"\b(?:physical|liquid|accidental)\s+damage\b",
    re.IGNORECASE,
)

REFUSAL_PHRASES = (
    "only help with shopease",
    "can only help",
    "only assist with shopease",
    "cannot help with",
    "can't help with",
    "cannot assist",
    "can't assist",
    "cannot provide",
    "cannot write",
    "cannot solve",
    "outside my scope",
    "outside what i can help",
    "not able to help",
    "not something i can help",
    "i cannot help",
    "i can't help",
)

PROMPT_LEAKAGE_PHRASES = (
    "you are the shopease customer support assistant",
    "use only the shopease policies and facts provided to you",
    "never invent order details, delivery dates, refund amounts, policies, or account information",
)

SPECIAL_TOKEN_FRAGMENTS = ("<|", "|>", "[inst]", "[/inst]", "im_start", "im_end")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "your",
    "you",
    "our",
    "are",
    "can",
    "will",
    "have",
    "has",
    "was",
    "were",
    "not",
    "but",
    "from",
    "they",
    "them",
    "their",
    "please",
    "shopeease",
    "shopease",
    "within",
    "after",
    "before",
    "than",
    "then",
    "also",
    "into",
    "such",
    "about",
    "only",
    "more",
    "when",
    "what",
    "which",
    "where",
    "would",
    "could",
    "should",
    "there",
    "here",
    "its",
    "it's",
    "been",
    "being",
    "does",
    "did",
    "doing",
    "how",
    "who",
    "why",
    "all",
    "any",
    "each",
    "both",
    "some",
    "most",
    "may",
    "might",
    "must",
    "shall",
}


def extract_assistant(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def jaccard(left: str, right: str) -> float:
    a = content_tokens(left)
    b = content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def response_validity(response: str, truncated: bool) -> tuple[bool, list[str]]:
    reasons = []
    stripped = response.strip()
    if not stripped:
        reasons.append("empty")
    if len(stripped) < 20:
        reasons.append("too_short")
    if len(re.findall(r"[A-Za-z]+", stripped)) < 5:
        reasons.append("too_few_words")
    if truncated:
        reasons.append("truncated_at_token_cap")
    lower = stripped.lower()
    if any(phrase in lower for phrase in PROMPT_LEAKAGE_PHRASES):
        reasons.append("prompt_leakage")
    if any(fragment in lower for fragment in SPECIAL_TOKEN_FRAGMENTS):
        reasons.append("special_token_leakage")
    return not reasons, reasons


def policy_keyword_covered(response: str, expected_category: str) -> bool:
    return any(
        keyword in response.lower()
        for keyword in COVERAGE_KEYWORDS.get(expected_category, [])
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]


def _negated(sentence: str) -> bool:
    lower = sentence.lower()
    return any(token in lower for token in NEGATION_TOKENS)


def _conditional(sentence: str) -> bool:
    lower = sentence.lower().strip()
    return (
        lower.startswith(CONDITIONAL_PREFIXES)
        or lower.startswith(ECHO_PREFIXES)
        or " if " in lower
        or any(phrase in lower for phrase in ECHO_PHRASES)
    )


def _allowed_numbers(*texts: str) -> set[str]:
    numbers = set()
    for text in texts:
        for match in re.finditer(r"\d[\d,]*", text):
            numbers.add(match.group(0).replace(",", ""))
    return numbers


def _allowed_order_ids(*texts: str) -> set[str]:
    ids = {value.upper() for value in GENERIC_ALLOWED_ORDER_IDS}
    for text in texts:
        ids.update(match.group(0).upper() for match in ORDER_ID_PATTERN.finditer(text))
    return ids


def hallucination_flags(
    response: str,
    user: str,
    reference: str,
    expected_category: str,
    policy_text: str,
) -> list[str]:
    flags: list[str] = []
    sentences = _sentences(response)

    allowed_ids = _allowed_order_ids(user, reference, policy_text)
    if any(
        match.group(0).upper() not in allowed_ids
        for match in ORDER_ID_PATTERN.finditer(response)
    ):
        flags.append("fabricated_order_id")

    if any(
        match.group(0).lower() not in ALLOWED_EMAILS
        for match in EMAIL_PATTERN.finditer(response)
    ):
        flags.append("fabricated_email")

    allowed_text = " ".join((user, reference, policy_text))
    for match in PHONE_PATTERN.finditer(response):
        if match.group(0) not in allowed_text:
            flags.append("fabricated_phone")
            break

    allowed_dates = set()
    for text in (user, reference, policy_text):
        for pattern in DATE_PATTERNS:
            allowed_dates.update(
                match.group(0).lower() for match in pattern.finditer(text)
            )
    for pattern in DATE_PATTERNS:
        if any(
            match.group(0).lower() not in allowed_dates
            for match in pattern.finditer(response)
        ):
            flags.append("fabricated_date")
            break

    allowed_numbers = _allowed_numbers(policy_text, user, reference)
    if any(
        match.group(1).replace(",", "") not in allowed_numbers
        for match in CURRENCY_PATTERN.finditer(response)
    ):
        flags.append("fabricated_amount")

    for flag, pattern in FLAG_PATTERNS.items():
        if any(
            pattern.search(sentence) and not _negated(sentence)
            for sentence in sentences
        ):
            flags.append(flag)

    if expected_category == "order_status":
        if any(
            ORDER_STATUS_SPECIFIC.search(sentence)
            and not _negated(sentence)
            and not _conditional(sentence)
            for sentence in sentences
        ):
            flags.append("specific_order_status_claim")

    if expected_category == "returns":
        lower = response.lower()
        if RETURN_WINDOW_MISMATCH.search(lower) and "48 hours" not in lower:
            flags.append("return_window_mismatch")

    if expected_category == "warranty":
        if any(
            WARRANTY_DAMAGE_CLAIM.search(sentence) and not _negated(sentence)
            for sentence in sentences
        ):
            flags.append("warranty_covers_physical_damage")

    return sorted(set(flags))


def category_signals(response: str, expected_category: str) -> dict:
    tag_category = parse_category(response)
    fallback_category = classify(response)
    combined_category = (
        tag_category if tag_category != "unknown" else fallback_category
    )
    return {
        "tag_category": tag_category,
        "fallback_category": fallback_category,
        "combined_category": combined_category,
        "tag_correct": tag_category == expected_category,
        "fallback_correct": fallback_category == expected_category,
        "combined_correct": combined_category == expected_category,
    }


def summarize_flags(rows: list[dict]) -> dict:
    counter = Counter()
    for row in rows:
        counter.update(row.get("heuristic_hallucination_flags", []))
    return dict(sorted(counter.items()))
