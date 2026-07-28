"""Tokenizer-bounded window planning and microbatch packing.

Character windows bound characters, not the token count that drives embedding
memory. Attention cost is quadratic in sequence length, so the same 4,096
characters cost wildly different amounts depending on how densely they tokenize:
ordinary source is ~984 tokens, a minified line is ~2,157, and embedding the
latter as one sequence adds ~1,172 MiB of resident memory against ~266 MiB for
the same characters split into token-bounded windows.

Everything here is pure and tokenizer-agnostic so the policy is testable without
loading a model. The only tokenizer contact is :func:`content_token_offsets`,
which reads an already-produced encoding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_TOKENS = 1024
DEFAULT_OVERLAP_TOKENS = 64
# Microbatch packing budget: item_count * longest_padded_tokens. Padding is to
# the longest member, so this bounds the padded matrix a batch materializes.
DEFAULT_MAX_TOKEN_PRODUCT = 4096
# A candidate never exceeds the extractor's character ceiling, so a window
# fan-out above this implies a pathological tokenization rather than dense code.
# It is a tripwire, not a working limit: 4,096 characters cannot tokenize to
# more than 4,096 tokens, which is five windows at the default budget.
MAX_WINDOWS_PER_CANDIDATE = 16


@dataclass(frozen=True)
class TokenWindow:
    """A token-bounded slice of one candidate, in candidate-relative characters."""

    start_char: int
    end_char: int
    token_count: int


def content_token_offsets(encoding: Any) -> list[tuple[int, int]]:
    """Return character spans for real tokens, dropping ``[CLS]``/``[SEP]``.

    *encoding* is anything with ``offsets`` and, optionally, a
    ``special_tokens_mask`` — a ``tokenizers.Encoding`` in production. Special
    tokens carry a ``(0, 0)`` span, so leaving them in would make every window
    appear to start at the beginning of the text.
    """
    offsets: list[tuple[int, int]] = [(int(start), int(end)) for start, end in encoding.offsets]
    mask = getattr(encoding, "special_tokens_mask", None)
    if mask is None:
        return offsets
    return [span for span, special in zip(offsets, mask, strict=True) if not special]


def plan_token_windows(
    offsets: Sequence[tuple[int, int]],
    *,
    text_length: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
) -> list[TokenWindow]:
    """Split a candidate into windows of at most *max_tokens* tokens.

    Windows are contiguous in characters and overlap by *overlap_tokens* tokens,
    so no source is dropped between them. Boundaries depend only on the
    tokenization, never on the memory budget or on how a batch was packed, so a
    retry at a smaller microbatch size re-derives the identical windows.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if text_length < 0:
        raise ValueError("text_length must not be negative")
    # Half the budget is the widest overlap that keeps the window count within
    # twice the minimum. Without it a budget squeezed by a wide prefix would
    # leave a one-token stride and fan a candidate out into hundreds of windows.
    overlap = min(max(overlap_tokens, 0), max_tokens - 1, max_tokens // 2)
    stride = max_tokens - overlap
    total = len(offsets)
    if total == 0:
        # Whitespace-only or untokenizable content still needs one window so the
        # candidate keeps a vector rather than silently vanishing from the index.
        return [TokenWindow(0, text_length, 0)] if text_length else []
    if total <= max_tokens:
        return [TokenWindow(0, text_length, total)]

    windows: list[TokenWindow] = []
    start_token = 0
    while start_token < total:
        end_token = min(start_token + max_tokens, total)
        # Character bounds run from this token's start to the *next* token's
        # start, so inter-token whitespace stays attached to the earlier window
        # and the concatenated windows cover the candidate exactly.
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = text_length if end_token == total else offsets[end_token][0]
        windows.append(TokenWindow(start_char, end_char, end_token - start_token))
        if len(windows) > max_windows:
            raise ValueError(
                f"Token planning exceeded {max_windows} windows for a "
                f"{text_length}-character candidate ({total} tokens)"
            )
        if end_token == total:
            break
        start_token += stride
    return windows


def plan_candidate_windows(
    encode: Callable[[str], Any],
    candidates: Sequence[tuple[str, str]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
) -> list[list[TokenWindow]]:
    """Plan windows for ``(prefix, content)`` candidates.

    The prefix is the context header repeated on every window of a candidate, so
    it is charged against the budget once and the content windows are sized with
    what is left.
    """
    plans: list[list[TokenWindow]] = []
    prefix_tokens: dict[str, int] = {}
    for prefix, content in candidates:
        if prefix not in prefix_tokens:
            prefix_tokens[prefix] = len(content_token_offsets(encode(prefix))) if prefix else 0
        # Keep at least one token of forward progress per window even when a
        # pathological prefix would otherwise consume the whole budget.
        budget = max(overlap_tokens + 1, max_tokens - prefix_tokens[prefix])
        plans.append(
            plan_token_windows(
                content_token_offsets(encode(content)),
                text_length=len(content),
                max_tokens=budget,
                overlap_tokens=overlap_tokens,
                max_windows=max_windows,
            )
        )
    return plans


def plan_microbatches(
    token_counts: Sequence[int],
    *,
    max_items: int = 1,
    max_token_product: int = DEFAULT_MAX_TOKEN_PRODUCT,
) -> list[list[int]]:
    """Bucket segment indices by length, then stay within both packing limits.

    A batch pads to its longest member, so ``item_count * longest`` — not the
    sum — is what the model materializes. A single segment always forms a batch
    even when it exceeds the product on its own; there is nothing smaller to
    fall back to. Power-of-two buckets keep similarly sized segments together
    without making exact token counts part of the ordering contract.
    """
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    ordered = sorted(
        enumerate(token_counts),
        key=lambda item: (
            -1 if item[1] > max_token_product else max(0, item[1]).bit_length(),
            item[0],
        ),
    )
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index, count in ordered:
        widened = max(longest, count)
        if current and (
            len(current) + 1 > max_items or (len(current) + 1) * widened > max_token_product
        ):
            batches.append(current)
            current = []
            widened = count
        current.append(index)
        longest = widened
    if current:
        batches.append(current)
    return batches
