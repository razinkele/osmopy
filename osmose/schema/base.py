"""Schema base classes for OSMOSE parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParamType(Enum):
    """Types of OSMOSE configuration parameters."""

    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    FILE_PATH = "file_path"
    MATRIX = "matrix"
    ENUM = "enum"


@dataclass
class OsmoseField:
    """Metadata for a single OSMOSE parameter.

    Attributes:
        key_pattern: OSMOSE property key, e.g. "species.linf.sp{idx}".
            Use {idx} placeholder for species-indexed parameters.
        param_type: The data type of this parameter.
        default: Default value if not specified.
        min_val: Minimum allowed value (numeric types only).
        max_val: Maximum allowed value (numeric types only).
        description: Human-readable description for UI tooltips.
        category: UI grouping category (e.g. "growth", "reproduction").
        unit: Physical unit (e.g. "cm", "year^-1").
        choices: Valid values for ENUM type.
        indexed: True if this parameter is per-species (uses sp{idx}).
        required: Whether this parameter must be specified.
        advanced: If True, shown only in the advanced config panel.
    """

    key_pattern: str
    param_type: ParamType
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    description: str = ""
    category: str = ""
    unit: str = ""
    choices: list[str] | None = None
    indexed: bool = False
    required: bool = True
    advanced: bool = False

    def __post_init__(self) -> None:
        if self.indexed and "{idx}" not in self.key_pattern:
            raise ValueError(
                f"OsmoseField indexed=True but key_pattern has no {{idx}}: {self.key_pattern}"
            )
        if not self.indexed and "{idx}" in self.key_pattern:
            raise ValueError(
                f"OsmoseField indexed=False but key_pattern contains {{idx}}: {self.key_pattern}"
            )
        if self.choices is not None and self.param_type != ParamType.ENUM:
            raise ValueError(
                f"OsmoseField has choices but param_type is "
                f"{self.param_type}, not ENUM: {self.key_pattern}"
            )
        if self.min_val is not None and self.max_val is not None and self.min_val > self.max_val:
            raise ValueError(
                f"OsmoseField min_val ({self.min_val}) > max_val "
                f"({self.max_val}): {self.key_pattern}"
            )

    def resolve_key(self, idx: int | None = None) -> str:
        """Resolve the key pattern to a concrete OSMOSE property key.

        Args:
            idx: Species index. Required for indexed fields, ignored for
                non-indexed fields.

        Returns:
            The resolved key string with {idx} replaced by the actual index.

        Raises:
            ValueError: If the field is indexed but no index is provided.
        """
        if self.indexed:
            if idx is None:
                raise ValueError(f"Index required for indexed field: {self.key_pattern}")
            return self.key_pattern.replace("{idx}", str(idx))
        return self.key_pattern

    def validate_value(self, value: Any) -> list[str]:
        """Validate a value against this field's constraints.

        Args:
            value: The value to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        if value is None:
            return errors
        if self.param_type in (ParamType.FLOAT, ParamType.INT):
            if self.min_val is not None and value < self.min_val:
                errors.append(f"Value {value} below min {self.min_val}")
            if self.max_val is not None and value > self.max_val:
                errors.append(f"Value {value} above max {self.max_val}")
        if self.param_type == ParamType.ENUM and self.choices:
            if value not in self.choices:
                errors.append(f"Value '{value}' not in choices: {self.choices}")
        return errors
