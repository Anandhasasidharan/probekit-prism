#!/usr/bin/env python3
"""Generate a reproducible, structurally labeled PII benchmark."""

import argparse
import json
import random
import re
from pathlib import Path

KINDS = ("email", "phone", "ssn", "card")
TEMPLATES = (
    "In the service desk record {ref}, the submitted value was {value}. Please verify the form.",
    "A customer added {value} to field {ref} while updating the account.",
    "The support ticket {ref} contains this entry: {value}. It came from a copied form.",
    "During review of request {ref}, staff wrote down {value} from the contact sheet.",
)
OOD_TEMPLATES = (
    "Warehouse ingest log [{ref}] :: barcode note captured <{value}> before packing.",
    "Transcript from the routing desk, item {ref}. Operator reads “{value}” over the radio.",
    "Archive docket {ref} — original inscription «{value}», retained with the exhibit inventory.",
    "Telemetry batch {ref} reported the following payload ({value}) during a cold start.",
)
PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[a-z0-9][a-z0-9.+-]*@[a-z0-9-]+\.(?:invalid|test)\b"),
    "phone": re.compile(r"(?<![\w-])(?:202|415)-555-01\d{2}(?![\w-])"),
    "ssn": re.compile(r"(?<![\w-])000-\d{2}-\d{4}(?![\w-])"),
    "card": re.compile(r"(?<![\w-])(?:\d{4}[ -]){3}\d{4}(?![\w-])"),
}


def luhn_valid(digits: str) -> bool:
    numbers = [int(number) for number in digits]
    for i in range(len(numbers) - 2, -1, -2):
        numbers[i] = numbers[i] * 2 - (9 if numbers[i] >= 5 else 0)
    return sum(numbers) % 10 == 0


def pii_kinds(text: str) -> set[str]:
    matches = {kind for kind in KINDS if kind != "card" and PATTERNS[kind].search(text)}
    if any(luhn_valid(re.sub(r"\D", "", match[0])) for match in PATTERNS["card"].finditer(text)):
        matches.add("card")
    return matches


def fake_value(rng: random.Random, kind: str, label: int, ood: bool) -> str:
    if kind == "email":
        mailbox = rng.choice(("gia", "ren", "mika", "lee", "tess", "kai"))
        domain = rng.choice(
            ("warehouse.test", "archive.test", "telemetry.test")
            if ood
            else ("desk.invalid", "intake.invalid", "support.invalid")
        )
        separator = "@" if label else rng.choice((" at ", " / ", " [at] "))
        return f"{mailbox}{rng.randrange(1000, 99999)}{separator}{domain}"
    if kind == "phone":
        area = "415" if ood else "202"
        middle = "555" if label else rng.choice(("55", "5555", "55X"))
        return f"{area}-{middle}-01{rng.randrange(100):02d}"
    if kind == "ssn":
        prefix = "000" if label else rng.choice(("00X", "0000", "00"))
        return f"{prefix}-{rng.randrange(100):02d}-{rng.randrange(10000):04d}"
    # 0000 is a deliberately invalid payment prefix. The test label is a Luhn-format proxy.
    digits = "0000" + str(rng.randrange(10**11)).zfill(11)
    check = next(d for d in range(10) if luhn_valid(digits + str(d)))
    digits += str(check if label else (check + rng.randrange(1, 10)) % 10)
    joiner = "-" if ood else " "
    return joiner.join(digits[i : i + 4] for i in range(0, 16, 4))


def make_rows(seed: int, sizes: dict[str, int]) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for split, size in sizes.items():
        if size < 8 or size % 8:
            raise ValueError("each split size must be a positive multiple of 8")
        ood = split == "ood"
        templates = OOD_TEMPLATES if ood else TEMPLATES
        conditions = [(kind, label) for kind in KINDS for label in (0, 1)] * (size // 8)
        rng.shuffle(conditions)
        for index, (kind, label) in enumerate(conditions):
            template_index = rng.randrange(len(templates))
            value = fake_value(rng, kind, label, ood)
            text = templates[template_index].format(ref=f"R{rng.getrandbits(64):016x}", value=value)
            if pii_kinds(text) != ({kind} if label else set()):
                raise ValueError(f"synthetic example violated its pattern label: {text}")
            rows.append(
                {
                    "sequence_id": f"{split}-{index:05d}",
                    "text": text,
                    "split": split,
                    "label": label,
                    "kind": kind,
                    "template_id": f"{'ood' if ood else 'id'}-{template_index}",
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--ood", type=int, default=400)
    args = parser.parse_args()
    sizes = {split: getattr(args, split) for split in ("train", "val", "test", "ood")}
    rows = make_rows(args.seed, sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
