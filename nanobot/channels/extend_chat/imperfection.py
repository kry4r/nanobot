"""Imperfection injector: adds humanlike typos and casual punctuation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import ImperfectionConfig


class ImperfectionInjector:
    """Post-processes text to add humanlike imperfections.

    Two mechanisms:
    - Homophone typo injection (~2.5% of characters)
    - Punctuation casualization (~15% of punctuation marks)
    """

    def __init__(self, config: "ImperfectionConfig"):
        self._config = config
        self._homophones: dict[str, list[str]] = self._load_homophones()

    def _load_homophones(self) -> dict[str, list[str]]:
        """Load homophone dictionary from config path or built-in."""
        path = self._config.homophone_dict_path
        if not path:
            path = str(Path(__file__).parent.parent.parent / "data" / "homophones_zh.json")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded {len(data)} homophone entries from {path}")
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load homophones from {path}: {e}")
            return {}

    def process(self, text: str) -> str:
        """Apply imperfections to a sentence."""
        if not self._config.post_processing:
            return text
        text = self._inject_homophones(text)
        text = self._casualize_punctuation(text)
        return text

    def _inject_homophones(self, text: str) -> str:
        """Replace characters with homophones at configured probability."""
        if not self._homophones or self._config.typo_probability <= 0:
            return text

        chars = list(text)
        for i, ch in enumerate(chars):
            if ch in self._homophones and random.random() < self._config.typo_probability:
                chars[i] = random.choice(self._homophones[ch])
        return "".join(chars)

    def _casualize_punctuation(self, text: str) -> str:
        """Casualize punctuation marks at configured probability."""
        if self._config.punctuation_casualness <= 0:
            return text

        replacements: dict[str, list[str]] = {
            "。": ["~", ".", "", "～"],
            "，": [",", " ", "~"],
            "！": ["!", "!!", "~", "！！"],
            "？": ["?", "??", "？？", "???"],
        }

        for formal, casual_options in replacements.items():
            if formal in text and random.random() < self._config.punctuation_casualness:
                text = text.replace(formal, random.choice(casual_options), 1)

        return text
