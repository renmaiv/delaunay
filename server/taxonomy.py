"""Loader/validator for shared/taxonomy.json — the single source of truth for
detection categories, tooltip copy, heatmap color bands, and the display
threshold. The frontend imports the same file at build time; never duplicate
its contents in code.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from server.schemas import DetectionCategory

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "shared" / "taxonomy.json"

_COLOR_RE = re.compile(r"^#[0-9a-f]{6}$")
_CATEGORY_KEYS = {"side", "source", "label", "short", "tooltip"}


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def validate_taxonomy(taxonomy: dict | None = None) -> None:
    """Raise ValueError describing the first problem found, else return None."""
    t = taxonomy if taxonomy is not None else load_taxonomy()

    categories = t.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("taxonomy: 'categories' must be an object")

    expected = {c.value for c in DetectionCategory}
    missing = expected - set(categories)
    if missing:
        raise ValueError(f"taxonomy: missing categories: {sorted(missing)}")
    extra = set(categories) - expected
    if extra:
        raise ValueError(f"taxonomy: unknown categories: {sorted(extra)}")

    for name, entry in categories.items():
        if not isinstance(entry, dict) or set(entry) != _CATEGORY_KEYS:
            raise ValueError(
                f"taxonomy: category '{name}' must have exactly keys {sorted(_CATEGORY_KEYS)}"
            )
        if entry["side"] not in ("user", "model"):
            raise ValueError(f"taxonomy: category '{name}' side must be 'user' or 'model'")
        for key in _CATEGORY_KEYS:
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"taxonomy: category '{name}' key '{key}' must be a non-empty string")

    bands = t.get("score_bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("taxonomy: 'score_bands' must be a non-empty list")
    if bands[0].get("min") != 0.0:
        raise ValueError("taxonomy: first score band must start at 0.0")
    if bands[-1].get("max") != 1.0:
        raise ValueError("taxonomy: last score band must end at 1.0")
    for i, band in enumerate(bands):
        if not _COLOR_RE.match(str(band.get("color", ""))):
            raise ValueError(f"taxonomy: score band {i} color must match #rrggbb")
        if i > 0 and bands[i - 1].get("max") != band.get("min"):
            raise ValueError(f"taxonomy: score bands not contiguous at index {i}")

    threshold = t.get("display_threshold")
    if not isinstance(threshold, (int, float)) or not (0.0 < threshold < 1.0):
        raise ValueError("taxonomy: 'display_threshold' must be a float in (0, 1)")
