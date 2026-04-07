"""Central registry for all OSMOSE parameters."""

from __future__ import annotations

import re

from osmose.schema.base import OsmoseField


class ParameterRegistry:
    """Collects all OSMOSE parameter definitions and provides lookup/validation.

    Thread-safety note: ``_match_cache`` is lazily populated and is NOT thread-safe.
    This is intentional — the app targets local single-worker deployment.  If running
    under a multi-worker Shiny server, wrap ``register`` and ``match`` calls with a
    ``threading.Lock``.  The engine's genetics module also calls ``register`` at
    runtime, so a global freeze is not appropriate.
    """

    def __init__(self):
        self._fields: list[OsmoseField] = []
        self._by_pattern: dict[str, OsmoseField] = {}
        self._compiled: list[tuple[re.Pattern, OsmoseField]] = []
        self._match_cache: dict[str, OsmoseField | None] = {}
        self._categories: list[str] = []

    def register(self, field: OsmoseField) -> None:
        self._fields.append(field)
        self._by_pattern[field.key_pattern] = field
        regex_str = re.escape(field.key_pattern).replace(r"\{idx\}", r"\d+")
        self._compiled.append((re.compile(regex_str), field))
        self._match_cache.clear()
        if field.category not in self._categories:
            self._categories.append(field.category)

    def all_fields(self) -> list[OsmoseField]:
        return list(self._fields)

    def fields_by_category(self, category: str) -> list[OsmoseField]:
        return [f for f in self._fields if f.category == category]

    def get_field(self, key_pattern: str) -> OsmoseField | None:
        return self._by_pattern.get(key_pattern)

    def categories(self) -> list[str]:
        return list(self._categories)

    def match_field(self, concrete_key: str) -> OsmoseField | None:
        """Match a concrete key like 'species.k.sp0' to its field pattern."""
        if concrete_key in self._match_cache:
            return self._match_cache[concrete_key]
        for compiled_re, field in self._compiled:
            if compiled_re.fullmatch(concrete_key):
                self._match_cache[concrete_key] = field
                return field
        self._match_cache[concrete_key] = None
        return None

    def validate(self, config: dict[str, object]) -> list[str]:
        """Validate a flat config dict against registered field constraints."""
        errors = []
        for key, value in config.items():
            field = self.match_field(key)
            if field:
                field_errors = field.validate_value(value)
                for e in field_errors:
                    errors.append(f"{key}: {e}")
        return errors
