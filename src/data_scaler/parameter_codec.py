from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from enum import Enum
from typing import TYPE_CHECKING, cast, get_type_hints

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
        """
        hints: dict[str, object] = get_type_hints(parameter_type)
        decoded: dict[str, object] = {
            name: ParameterCodec._decode_value(hints[name], value)
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
    def _decode_value(hint: object, value: JsonValue) -> object:
        if isinstance(hint, type) and issubclass(hint, Enum):
            return hint(value)
        if isinstance(value, list):
            return tuple(value)
        return value
