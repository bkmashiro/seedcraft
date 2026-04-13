"""Schema definitions for Seedcraft data generation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union


class FieldType(enum.Enum):
    """Supported field types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    CHOICE = "choice"
    COMPOSITE = "composite"


class DistributionKind(enum.Enum):
    """Distribution families for numeric generation."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    POISSON = "poisson"
    ZIPF = "zipf"


@dataclass
class Distribution:
    """Describes how values for a field are distributed.

    Parameters
    ----------
    kind : DistributionKind
        The distribution family.
    params : dict
        Distribution-specific parameters (e.g. mean, std, alpha, beta).
    """
    kind: DistributionKind
    params: dict[str, float] = field(default_factory=dict)

    # Convenience constructors
    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0) -> Distribution:
        return cls(DistributionKind.UNIFORM, {"low": low, "high": high})

    @classmethod
    def normal(cls, mean: float = 0.0, std: float = 1.0) -> Distribution:
        return cls(DistributionKind.NORMAL, {"mean": mean, "std": std})

    @classmethod
    def lognormal(cls, mean: float = 0.0, sigma: float = 1.0) -> Distribution:
        return cls(DistributionKind.LOGNORMAL, {"mean": mean, "sigma": sigma})

    @classmethod
    def exponential(cls, scale: float = 1.0) -> Distribution:
        return cls(DistributionKind.EXPONENTIAL, {"scale": scale})

    @classmethod
    def beta(cls, alpha: float = 2.0, beta: float = 5.0) -> Distribution:
        return cls(DistributionKind.BETA, {"alpha": alpha, "beta": beta})

    @classmethod
    def poisson(cls, lam: float = 5.0) -> Distribution:
        return cls(DistributionKind.POISSON, {"lam": lam})

    @classmethod
    def zipf(cls, a: float = 2.0) -> Distribution:
        return cls(DistributionKind.ZIPF, {"a": a})


@dataclass
class Field:
    """A single field in a schema.

    Parameters
    ----------
    name : str
        The field name (column name in output).
    field_type : FieldType
        The data type.
    distribution : Distribution | None
        How values are distributed (for numeric/date types).
    choices : list | None
        Valid values for CHOICE fields.  Can be a list of values or
        a list of (value, weight) tuples for weighted selection.
    nullable : float
        Probability that a value is null (0.0 = never, 1.0 = always).
    min_val : float | None
        Hard minimum (clamp).
    max_val : float | None
        Hard maximum (clamp).
    format_str : str | None
        Format string for dates ("%Y-%m-%d") or string templates ("{prefix}-{seq:05d}").
    unique : bool
        If True, generated values are deduplicated.
    transform : Callable | None
        Post-generation transform applied to each value.
    prefix : str | None
        For STRING composite fields, a prefix.
    """
    name: str
    field_type: FieldType = FieldType.FLOAT
    distribution: Optional[Distribution] = None
    choices: Optional[Sequence[Any]] = None
    nullable: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    format_str: Optional[str] = None
    unique: bool = False
    transform: Optional[Callable[[Any], Any]] = None
    prefix: Optional[str] = None

    # Convenience constructors -------------------------------------------------

    @classmethod
    def integer(cls, name: str, low: int = 0, high: int = 100, **kwargs) -> Field:
        if "distribution" not in kwargs:
            kwargs["distribution"] = Distribution.uniform(low, high)
        return cls(
            name=name,
            field_type=FieldType.INTEGER,
            min_val=kwargs.pop("min_val", low),
            max_val=kwargs.pop("max_val", high),
            **kwargs,
        )

    @classmethod
    def real(cls, name: str, distribution: Optional[Distribution] = None, **kwargs) -> Field:
        return cls(
            name=name,
            field_type=FieldType.FLOAT,
            distribution=distribution or Distribution.normal(),
            **kwargs,
        )

    @classmethod
    def choice(cls, name: str, options: Sequence[Any], **kwargs) -> Field:
        return cls(name=name, field_type=FieldType.CHOICE, choices=options, **kwargs)

    @classmethod
    def boolean(cls, name: str, true_probability: float = 0.5, **kwargs) -> Field:
        return cls(
            name=name,
            field_type=FieldType.BOOLEAN,
            distribution=Distribution.uniform(0, 1),
            min_val=true_probability,  # overloaded: threshold
            **kwargs,
        )

    @classmethod
    def date(cls, name: str, start: str = "2020-01-01", end: str = "2025-12-31",
             fmt: str = "%Y-%m-%d", **kwargs) -> Field:
        return cls(
            name=name,
            field_type=FieldType.DATE,
            format_str=fmt,
            distribution=Distribution.uniform(),
            min_val=0,  # will be mapped to start
            max_val=1,  # will be mapped to end
            prefix=f"{start}|{end}",  # encode range
            **kwargs,
        )

    @classmethod
    def string_id(cls, name: str, prefix: str = "ID", **kwargs) -> Field:
        return cls(
            name=name,
            field_type=FieldType.STRING,
            prefix=prefix,
            unique=True,
            **kwargs,
        )


class CorrelationType(enum.Enum):
    """Types of inter-field correlations."""
    LINEAR = "linear"          # y ≈ a*x + b + noise
    MONOTONIC = "monotonic"    # y increases/decreases with x
    CONDITIONAL = "conditional"  # y depends on category of x
    DERIVED = "derived"        # y = f(x) exactly
    MUTUAL_EXCLUSIVE = "mutual_exclusive"  # if x then not y


@dataclass
class Correlation:
    """Defines a relationship between two fields.

    Parameters
    ----------
    source : str
        Name of the source (independent) field.
    target : str
        Name of the target (dependent) field.
    correlation_type : CorrelationType
        The kind of correlation.
    strength : float
        How strong the correlation is (0.0 = none, 1.0 = perfect).
    params : dict
        Type-specific parameters:
        - LINEAR: {"slope": float, "intercept": float}
        - CONDITIONAL: {"mapping": {source_value: Distribution}}
        - DERIVED: {"func": Callable}
    """
    source: str
    target: str
    correlation_type: CorrelationType = CorrelationType.LINEAR
    strength: float = 0.8
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def linear(cls, source: str, target: str, strength: float = 0.8,
               slope: float = 1.0, intercept: float = 0.0) -> Correlation:
        return cls(source, target, CorrelationType.LINEAR, strength,
                   {"slope": slope, "intercept": intercept})

    @classmethod
    def monotonic(cls, source: str, target: str, strength: float = 0.8,
                  increasing: bool = True) -> Correlation:
        return cls(source, target, CorrelationType.MONOTONIC, strength,
                   {"increasing": increasing})

    @classmethod
    def conditional(cls, source: str, target: str,
                    mapping: dict[Any, Distribution]) -> Correlation:
        return cls(source, target, CorrelationType.CONDITIONAL, 1.0,
                   {"mapping": mapping})

    @classmethod
    def derived(cls, source: str, target: str,
                func: Callable[[Any], Any]) -> Correlation:
        return cls(source, target, CorrelationType.DERIVED, 1.0,
                   {"func": func})

    @classmethod
    def mutual_exclusive(cls, source: str, target: str) -> Correlation:
        return cls(source, target, CorrelationType.MUTUAL_EXCLUSIVE, 1.0)


@dataclass
class Schema:
    """A complete data generation schema.

    Contains field definitions and inter-field correlations.
    Schemas can be composed and nested.
    """
    name: str
    fields: list[Field] = field(default_factory=list)
    correlations: list[Correlation] = field(default_factory=list)
    row_count: int = 100

    def add_field(self, f: Field) -> Schema:
        """Add a field and return self for chaining."""
        self.fields.append(f)
        return self

    def add_correlation(self, c: Correlation) -> Schema:
        """Add a correlation and return self for chaining."""
        self.correlations.append(c)
        return self

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get_field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"Field '{name}' not found in schema '{self.name}'")

    def validate(self) -> list[str]:
        """Validate the schema, returning a list of error messages."""
        errors: list[str] = []
        names = self.field_names()

        # Check for duplicate field names
        seen = set()
        for n in names:
            if n in seen:
                errors.append(f"Duplicate field name: '{n}'")
            seen.add(n)

        # Check correlations reference valid fields
        for c in self.correlations:
            if c.source not in names:
                errors.append(f"Correlation source '{c.source}' not in schema fields")
            if c.target not in names:
                errors.append(f"Correlation target '{c.target}' not in schema fields")

        # Check choice fields have choices
        for f in self.fields:
            if f.field_type == FieldType.CHOICE and not f.choices:
                errors.append(f"Choice field '{f.name}' has no choices defined")

        return errors

    def to_dict(self) -> dict:
        """Serialize schema to a dictionary (excluding callables)."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "fields": [
                {
                    "name": f.name,
                    "type": f.field_type.value,
                    "nullable": f.nullable,
                    "unique": f.unique,
                    "min_val": f.min_val,
                    "max_val": f.max_val,
                }
                for f in self.fields
            ],
            "correlations": [
                {
                    "source": c.source,
                    "target": c.target,
                    "type": c.correlation_type.value,
                    "strength": c.strength,
                }
                for c in self.correlations
            ],
        }
