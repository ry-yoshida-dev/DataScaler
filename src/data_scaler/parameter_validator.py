from __future__ import annotations

import sys
from dataclasses import fields
from typing import TYPE_CHECKING, cast, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class ParameterValidator:
    """
    Checks parameter fields against their annotations, which Python does not enforce.
    """

    @staticmethod
    def require_valid_fields(parameters: DataclassInstance) -> None:
        """
        Check that every field has its annotated type and a representable value.

        Parameters
        ----------
        parameters : DataclassInstance
            The parameters.

        Raises
        ------
        TypeError
            If a field does not have its annotated type.
        ValueError
            If a number in a tuple field is too large for a float.
        """
        hints: dict[str, object] = get_type_hints(type(parameters))
        for field in fields(parameters):
            value: object = cast(object, getattr(parameters, field.name))
            if not ParameterValidator._matches(hints[field.name], value):
                raise TypeError(
                    f"{field.name} must be {hints[field.name]}, got {value!r}"
                )
            if isinstance(value, tuple) and any(
                abs(item) > sys.float_info.max
                for item in cast(tuple[float, ...], value)
            ):
                raise ValueError(f"{field.name} must fit a float, got {value!r}")

    @staticmethod
    def _matches(hint: object, value: object) -> bool:
        if get_origin(hint) is tuple:
            item_hints: tuple[object, ...] = get_args(hint)
            return (
                isinstance(value, tuple)
                and len(cast(tuple[object, ...], value)) == len(item_hints)
                and all(
                    ParameterValidator._matches(item_hint, item)
                    for item_hint, item in zip(
                        item_hints, cast(tuple[object, ...], value), strict=True
                    )
                )
            )
        if hint is float:
            return isinstance(value, int | float) and not isinstance(value, bool)
        return type(value) in (get_args(hint) or (hint,))
