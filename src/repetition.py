from collections import Counter


SLOT_NAMES = ("H", "T", "O")


def compute_r_value_from_tokens(tokens):
    """Compute the proportion of unordered token pairs that contain identical tokens."""
    tokens = list(tokens)
    if len(tokens) < 2:
        return 0.0
    equal_pairs = sum(count * (count - 1) // 2 for count in Counter(tokens).values())
    total_pairs = len(tokens) * (len(tokens) - 1) // 2
    return equal_pairs / total_pairs


def compute_slot_repetition(tokens, digits_per_value=3):
    """Compute global pairwise repetition rates for each digit slot (H/T/O)."""
    return {
        slot_name: compute_r_value_from_tokens([token[slot] for token in tokens])
        for slot, slot_name in enumerate(SLOT_NAMES[:digits_per_value])
    }
