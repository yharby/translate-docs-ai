"""
Font management and fallback system for cross-platform compatibility.

Provides intelligent font fallback for RTL (Arabic, Hebrew) and LTR languages
using fonts available on all major operating systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FontCategory(str, Enum):
    """Font categories for fallback selection."""

    SERIF = "serif"
    SANS_SERIF = "sans-serif"
    MONOSPACE = "monospace"
    DISPLAY = "display"
    HANDWRITING = "handwriting"


class ScriptType(str, Enum):
    """Script types for language support."""

    LATIN = "latin"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    CJK = "cjk"  # Chinese, Japanese, Korean
    CYRILLIC = "cyrillic"
    GREEK = "greek"
    THAI = "thai"
    DEVANAGARI = "devanagari"


# Language to script mapping
LANGUAGE_SCRIPTS: dict[str, ScriptType] = {
    "en": ScriptType.LATIN,
    "fr": ScriptType.LATIN,
    "de": ScriptType.LATIN,
    "es": ScriptType.LATIN,
    "it": ScriptType.LATIN,
    "pt": ScriptType.LATIN,
    "ar": ScriptType.ARABIC,
    "fa": ScriptType.ARABIC,  # Persian/Farsi
    "ur": ScriptType.ARABIC,  # Urdu
    "he": ScriptType.HEBREW,
    "zh": ScriptType.CJK,
    "ja": ScriptType.CJK,
    "ko": ScriptType.CJK,
    "ru": ScriptType.CYRILLIC,
    "el": ScriptType.GREEK,
    "th": ScriptType.THAI,
    "hi": ScriptType.DEVANAGARI,
}

# RTL scripts
RTL_SCRIPTS = {ScriptType.ARABIC, ScriptType.HEBREW}


@dataclass
class FontFallback:
    """Font fallback configuration."""

    primary: str
    fallbacks: list[str]
    category: FontCategory
    supports_scripts: set[ScriptType]


# Cross-platform font fallbacks
# These fonts are available on Windows, macOS, and most Linux distributions
FONT_FALLBACKS: dict[str, FontFallback] = {
    # Sans-serif fonts with broad language support
    "Arial": FontFallback(
        primary="Arial",
        fallbacks=["Tahoma", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        category=FontCategory.SANS_SERIF,
        supports_scripts={
            ScriptType.LATIN,
            ScriptType.CYRILLIC,
            ScriptType.GREEK,
            ScriptType.ARABIC,
        },
    ),
    "Helvetica": FontFallback(
        primary="Helvetica",
        fallbacks=["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        category=FontCategory.SANS_SERIF,
        supports_scripts={ScriptType.LATIN, ScriptType.CYRILLIC, ScriptType.GREEK},
    ),
    # Serif fonts
    "Times New Roman": FontFallback(
        primary="Times New Roman",
        fallbacks=["Times", "Liberation Serif", "DejaVu Serif", "serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.LATIN, ScriptType.CYRILLIC, ScriptType.GREEK},
    ),
    "Georgia": FontFallback(
        primary="Georgia",
        fallbacks=["Times New Roman", "Liberation Serif", "serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.LATIN},
    ),
    # Monospace fonts
    "Courier New": FontFallback(
        primary="Courier New",
        fallbacks=["Courier", "Liberation Mono", "DejaVu Sans Mono", "monospace"],
        category=FontCategory.MONOSPACE,
        supports_scripts={ScriptType.LATIN, ScriptType.CYRILLIC},
    ),
    "Consolas": FontFallback(
        primary="Consolas",
        fallbacks=["Monaco", "Liberation Mono", "DejaVu Sans Mono", "monospace"],
        category=FontCategory.MONOSPACE,
        supports_scripts={ScriptType.LATIN},
    ),
    # Arabic fonts (cross-platform)
    "Tahoma": FontFallback(
        primary="Tahoma",
        fallbacks=["Arial", "Segoe UI", "sans-serif"],
        category=FontCategory.SANS_SERIF,
        supports_scripts={ScriptType.ARABIC, ScriptType.LATIN, ScriptType.HEBREW},
    ),
    "Segoe UI": FontFallback(
        primary="Segoe UI",
        fallbacks=["Tahoma", "Arial", "sans-serif"],
        category=FontCategory.SANS_SERIF,
        supports_scripts={ScriptType.ARABIC, ScriptType.LATIN, ScriptType.HEBREW},
    ),
    # Traditional Arabic fonts
    "Traditional Arabic": FontFallback(
        primary="Traditional Arabic",
        fallbacks=["Arabic Typesetting", "Tahoma", "Arial", "sans-serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.ARABIC},
    ),
    "Arabic Typesetting": FontFallback(
        primary="Arabic Typesetting",
        fallbacks=["Traditional Arabic", "Tahoma", "Arial", "sans-serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.ARABIC},
    ),
    # Hebrew fonts
    "David": FontFallback(
        primary="David",
        fallbacks=["Tahoma", "Arial", "sans-serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.HEBREW},
    ),
    # CJK fonts
    "SimSun": FontFallback(
        primary="SimSun",
        fallbacks=["MS Mincho", "Hiragino Mincho Pro", "serif"],
        category=FontCategory.SERIF,
        supports_scripts={ScriptType.CJK},
    ),
    "Microsoft YaHei": FontFallback(
        primary="Microsoft YaHei",
        fallbacks=["SimHei", "Hiragino Sans", "sans-serif"],
        category=FontCategory.SANS_SERIF,
        supports_scripts={ScriptType.CJK},
    ),
}

# Default fonts by script type (universally available)
DEFAULT_FONTS_BY_SCRIPT: dict[ScriptType, dict[FontCategory, str]] = {
    ScriptType.LATIN: {
        FontCategory.SERIF: "Times New Roman",
        FontCategory.SANS_SERIF: "Arial",
        FontCategory.MONOSPACE: "Courier New",
    },
    ScriptType.ARABIC: {
        FontCategory.SERIF: "Traditional Arabic",
        FontCategory.SANS_SERIF: "Tahoma",
        FontCategory.MONOSPACE: "Courier New",
    },
    ScriptType.HEBREW: {
        FontCategory.SERIF: "David",
        FontCategory.SANS_SERIF: "Tahoma",
        FontCategory.MONOSPACE: "Courier New",
    },
    ScriptType.CJK: {
        FontCategory.SERIF: "SimSun",
        FontCategory.SANS_SERIF: "Microsoft YaHei",
        FontCategory.MONOSPACE: "MS Gothic",
    },
    ScriptType.CYRILLIC: {
        FontCategory.SERIF: "Times New Roman",
        FontCategory.SANS_SERIF: "Arial",
        FontCategory.MONOSPACE: "Courier New",
    },
    ScriptType.GREEK: {
        FontCategory.SERIF: "Times New Roman",
        FontCategory.SANS_SERIF: "Arial",
        FontCategory.MONOSPACE: "Courier New",
    },
}


def get_script_for_language(language: str) -> ScriptType:
    """Get the script type for a language code."""
    return LANGUAGE_SCRIPTS.get(language, ScriptType.LATIN)


def is_rtl_language(language: str) -> bool:
    """Check if a language uses RTL script."""
    script = get_script_for_language(language)
    return script in RTL_SCRIPTS


def categorize_font(font_name: str) -> FontCategory:
    """Categorize a font based on its name."""
    font_lower = font_name.lower()

    # Monospace indicators
    if any(kw in font_lower for kw in ["mono", "courier", "consolas", "code", "fixed"]):
        return FontCategory.MONOSPACE

    # Serif indicators
    if any(
        kw in font_lower for kw in ["times", "georgia", "garamond", "palatino", "serif", "roman"]
    ):
        return FontCategory.SERIF

    # Sans-serif indicators (or default)
    if any(kw in font_lower for kw in ["arial", "helvetica", "verdana", "tahoma", "segoe", "sans"]):
        return FontCategory.SANS_SERIF

    # Default to sans-serif (most common for documents)
    return FontCategory.SANS_SERIF


def get_fallback_font(
    original_font: str | None,
    target_language: str,
    category: FontCategory | None = None,
) -> str:
    """
    Get a fallback font that supports the target language.

    Args:
        original_font: Original font name from source document.
        target_language: Target language code (en, ar, he, etc.).
        category: Font category (serif, sans-serif, monospace).

    Returns:
        Font name that supports the target language.
    """
    script = get_script_for_language(target_language)

    # Determine category from original font if not specified
    if category is None and original_font:
        category = categorize_font(original_font)
    elif category is None:
        category = FontCategory.SANS_SERIF

    # Check if original font supports the script
    if original_font and original_font in FONT_FALLBACKS:
        fallback = FONT_FALLBACKS[original_font]
        if script in fallback.supports_scripts:
            return original_font

    # Get default font for script and category
    script_defaults = DEFAULT_FONTS_BY_SCRIPT.get(script, DEFAULT_FONTS_BY_SCRIPT[ScriptType.LATIN])
    return script_defaults.get(category, script_defaults[FontCategory.SANS_SERIF])


class FontManager:
    """
    Manages font selection and fallback for document export.

    Provides intelligent font mapping that:
    - Preserves original font style (serif, sans-serif, monospace)
    - Ensures language support for target scripts
    - Uses cross-platform available fonts
    """

    def __init__(self, target_language: str = "en"):
        """
        Initialize font manager.

        Args:
            target_language: Target language code for font selection.
        """
        self.target_language = target_language
        self.script = get_script_for_language(target_language)
        self.is_rtl = self.script in RTL_SCRIPTS
        self._font_cache: dict[str, str] = {}

    def get_font(
        self,
        original_font: str | None,
        category: FontCategory | None = None,
    ) -> str:
        """
        Get appropriate font for the target language.

        Args:
            original_font: Original font from source document.
            category: Font category override.

        Returns:
            Font name suitable for the target language.
        """
        cache_key = f"{original_font}:{category}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        result = get_fallback_font(original_font, self.target_language, category)
        self._font_cache[cache_key] = result
        return result

    def get_font_family_css(
        self,
        original_font: str | None,
        category: FontCategory | None = None,
    ) -> str:
        """
        Get CSS font-family string with fallbacks.

        Args:
            original_font: Original font from source document.
            category: Font category override.

        Returns:
            CSS font-family value with fallback chain.
        """
        primary = self.get_font(original_font, category)

        # Build fallback chain
        if primary in FONT_FALLBACKS:
            fallback = FONT_FALLBACKS[primary]
            fonts = [primary] + fallback.fallbacks
        else:
            fonts = [primary, "sans-serif"]

        # Quote font names with spaces
        quoted = [f'"{f}"' if " " in f else f for f in fonts]
        return ", ".join(quoted)

    def get_default_fonts(self) -> dict[FontCategory, str]:
        """Get default fonts for all categories in target language."""
        script_defaults = DEFAULT_FONTS_BY_SCRIPT.get(
            self.script, DEFAULT_FONTS_BY_SCRIPT[ScriptType.LATIN]
        )
        return dict(script_defaults)
