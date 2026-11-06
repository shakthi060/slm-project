from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.policy_data import POLICY_DOCUMENT
from src.evaluation.baseline import classify

SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

SECTION_ALIASES = {
    "order_status": ("Orders",),
    "shipping": ("Shipping and delivery",),
    "returns": ("Returns",),
    "refunds": ("Refunds",),
    "payments": ("Payments",),
    "cancellations": ("Cancellations",),
    "product_info": ("Products",),
    "warranty": ("Warranty",),
    "account": ("Account support",),
    "complaints": ("Complaints and escalation",),
    "general_faq": ("General FAQ",),
    "clarification": ("Orders", "Returns", "Shipping and delivery"),
    "out_of_scope": ("Out-of-scope behaviour",),
}

ALWAYS_INCLUDE = ("Company",)

POLICY_INTENT_OVERRIDES = (
    (
        "product_info",
        (
            "price match",
            "price matching",
            "in stock",
            "out of stock",
            "stock availability",
            "product stock",
            "restock",
        ),
    ),
    (
        "account",
        (
            "change my phone number",
            "update my phone number",
            "change phone number",
            "update phone number",
            "phone number on my account",
        ),
    ),
)

OUT_OF_SCOPE_BLOCKERS = (
    "another customer",
    "someone else",
    "other customer",
    "system prompt",
    "ignore your rules",
    "hidden instructions",
)


def classify_request(message: str) -> str:
    """Use the shared classifier, with narrow fixes for known policy phrasing."""
    category = classify(message)
    if category != "out_of_scope":
        return category

    normalized = " ".join(message.lower().split())
    if any(blocker in normalized for blocker in OUT_OF_SCOPE_BLOCKERS):
        return category

    for target, phrases in POLICY_INTENT_OVERRIDES:
        if any(phrase in normalized for phrase in phrases):
            return target
    return category


@dataclass(frozen=True)
class PolicyContext:
    category: str
    section_names: tuple[str, ...]
    text: str
    found: bool
    source: str


def parse_policy_sections(document: str) -> dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(document))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        body = document[match.end() : end].strip()
        if body:
            sections[name] = f"## {name}\n\n{body}"
    return sections


class PolicyRetriever:
    """Retrieves only the policy sections relevant to a support category."""

    def __init__(self, document: str | None = None) -> None:
        self.document = document if document is not None else POLICY_DOCUMENT
        self.sections = parse_policy_sections(self.document)
        self.source = "src.data.policy_data.POLICY_DOCUMENT"

    def retrieve(
        self,
        message: str,
        category: str | None = None,
    ) -> PolicyContext:
        category = category or classify_request(message)
        requested = SECTION_ALIASES.get(category, ())
        matched = [name for name in requested if name in self.sections]
        if not matched:
            return PolicyContext(
                category=category,
                section_names=(),
                text="",
                found=False,
                source=self.source,
            )

        include = list(ALWAYS_INCLUDE) + [
            name for name in matched if name not in ALWAYS_INCLUDE
        ]
        text = "\n\n".join(
            self.sections[name] for name in include if name in self.sections
        )
        return PolicyContext(
            category=category,
            section_names=tuple(include),
            text=text,
            found=True,
            source=self.source,
        )
