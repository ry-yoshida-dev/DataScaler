from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from enum import Enum
from typing import TYPE_CHECKING, cast, get_args, get_origin, get_type_hints

from .types import JsonValue

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class ParameterCodec:
    """
    Converts parameter dataclasses to and from JSON-compatible values.
    """

    @staticmethod
    def encode(parameters: DataclassInstance) -> dict[str, JsonValue]:
        """
        Convert parameters to JSON-compatible values.

        Parameters
        ----------
        parameters : DataclassInstance
            The parameters.

        Returns
        -------
        dict[str, JsonValue]
            The values by field name.
        """
        return {
            field.name: ParameterCodec._encode_value(
                cast(object, getattr(parameters, field.name))
            )
            for field in fields(parameters)
        }

    @staticmethod
    def decode[ParametersT: DataclassInstance](
        parameter_type: type[ParametersT], encoded: Mapping[str, JsonValue]
    ) -> ParametersT:
        """
        Rebuild parameters, which validates the values again.

        Parameters
        ----------
        parameter_type : type[ParametersT]
            The concrete parameter class, whose fields all have defaults.
        encoded : Mapping[str, JsonValue]
            The values by field name.

        Returns
        -------
        ParametersT
            The parameters.

        Raises
        ------
        ValueError
            If a field is unknown or a value does not fit its field.
        """
        hints: dict[str, object] = get_type_hints(parameter_type)
        unknown: set[str] = set(encoded) - set(hints)
        if unknown:
            raise ValueError(
                f"Unknown fields of {parameter_type.__name__}: {sorted(unknown)}"
            )
        decoded: dict[str, object] = {
            name: ParameterCodec._decode_value(name, hints[name], value)
            for name, value in encoded.items()
        }
        return replace(parameter_type(), **decoded)

    @staticmethod
    def _encode_value(value: object) -> JsonValue:
        if isinstance(value, Enum):
            return cast(JsonValue, value.value)
        if isinstance(value, tuple):
            return list(cast(tuple[float, ...], value))
        return cast(JsonValue, value)

    @staticmethod
    def _decode_value(name: str, hint: object, value: JsonValue) -> object:
        if isinstance(hint, type) and issubclass(hint, Enum):
            return hint(value)
        if get_origin(hint) is tuple:
            if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, int | float)
                for item in value
            ):
                raise ValueError(f"{name} must be a list of numbers, got {value!r}")
            return tuple(value)
        if type(value) not in (get_args(hint) or (hint,)):
            raise ValueError(f"{name} has an invalid value: {value!r}")
        return value
