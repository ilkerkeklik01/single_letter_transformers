"""Create positive name/letter pairs without requiring a labelled dataset."""

import random


TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"

# These are teaching labels, not claims of perfect linguistic equivalence. They
# give the tiny model a few explicit letter relationships alongside name edits.
LETTER_PAIRS = (
    ("c", "ç"), ("g", "ğ"), ("g", "k"), ("s", "ş"), ("s", "z"),
    ("o", "ö"), ("u", "ü"), ("ı", "i"), ("b", "p"), ("d", "t"),
    ("m", "n"), ("l", "r"),
)

ASCII_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})


def corrupt_name(name: str, rng: random.Random) -> str:
    """Make one realistic spelling/noise view while keeping at least one char."""
    if not name:
        return name

    operation = rng.choice(("delete", "swap", "replace", "ascii", "identity"))
    chars = list(name)

    if operation == "delete" and len(chars) > 1:
        del chars[rng.randrange(len(chars))]
    elif operation == "swap" and len(chars) > 1:
        index = rng.randrange(len(chars) - 1)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    elif operation == "replace":
        index = rng.randrange(len(chars))
        source = chars[index]
        alternatives = [right for left, right in LETTER_PAIRS if left == source]
        alternatives += [left for left, right in LETTER_PAIRS if right == source]
        chars[index] = rng.choice(alternatives or TURKISH_LETTERS)
    elif operation == "ascii":
        return name.translate(ASCII_FOLD)

    return "".join(chars)


def make_positive_pairs(
    names: list[str], batch_size: int, rng: random.Random, letter_fraction: float = 0.125
) -> tuple[list[str], list[str]]:
    """Mix noisy-name positives with explicitly related single-letter positives."""
    anchors: list[str] = []
    positives: list[str] = []
    letter_count = min(len(LETTER_PAIRS), round(batch_size * letter_fraction))

    for left, right in rng.sample(LETTER_PAIRS, k=letter_count):
        if rng.random() < 0.5:
            left, right = right, left
        anchors.append(left)
        positives.append(right)

    for name in rng.sample(names, k=batch_size - letter_count):
        anchors.append(name)
        positives.append(corrupt_name(name, rng))

    order = list(range(batch_size))
    rng.shuffle(order)
    return [anchors[i] for i in order], [positives[i] for i in order]
