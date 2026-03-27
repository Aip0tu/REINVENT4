"""Minimal local fallback for typed-argument-parser.

This covers the small subset of Tap used by the bundled FLAME_plugin inference
code so prediction can run in environments where the external dependency is
not installed.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin


class Tap:
    def __init__(self, *args, **kwargs):
        self._apply_class_defaults()

    def add_argument(self, *args, **kwargs) -> None:
        # The FLAME inference path only uses this to declare choices.
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def from_dict(self, values: dict[str, Any], skip_unsettable: bool = False):
        for key, value in values.items():
            if skip_unsettable and not hasattr(self, key):
                continue
            descriptor = getattr(type(self), key, None)
            if skip_unsettable and isinstance(descriptor, property) and descriptor.fset is None:
                continue
            setattr(self, key, value)
        return self

    def parse_args(self, args: list[str] | None = None):
        if args is None:
            args = []

        self.configure()
        annotations = self._collect_annotations()
        index = 0

        while index < len(args):
            token = args[index]
            if not token.startswith("--"):
                index += 1
                continue

            key = token[2:].replace("-", "_")
            annotation = annotations.get(key, str)

            next_index = index + 1
            if self._is_bool(annotation):
                value = True
                if next_index < len(args) and not args[next_index].startswith("--"):
                    value = self._convert(args[next_index], annotation)
                    index += 1
            else:
                if next_index >= len(args):
                    raise ValueError(f"Missing value for argument {token}")
                value = self._convert(args[next_index], annotation)
                index += 1

            setattr(self, key, value)
            index += 1

        self.process_args()
        return self

    def configure(self) -> None:
        return None

    def process_args(self) -> None:
        return None

    def _apply_class_defaults(self) -> None:
        for cls in reversed(type(self).mro()):
            if cls is object:
                continue
            for key, value in vars(cls).items():
                if key.startswith("_") or callable(value) or isinstance(value, property):
                    continue
                setattr(self, key, value)

        for key in self._collect_annotations():
            if hasattr(self, key):
                continue
            setattr(self, key, None)

    def _collect_annotations(self) -> dict[str, Any]:
        annotations: dict[str, Any] = {}
        for cls in reversed(type(self).mro()):
            annotations.update(getattr(cls, "__annotations__", {}))
        return annotations

    def _is_bool(self, annotation: Any) -> bool:
        if annotation is bool:
            return True

        origin = get_origin(annotation)
        if origin is None:
            return False

        return any(arg is bool for arg in get_args(annotation))

    def _convert(self, value: str, annotation: Any) -> Any:
        if value == "None":
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (list, tuple):
            inner = args[0] if args else str
            return [self._convert(value, inner)]

        if origin is not None and args:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return self._convert(value, non_none[0])

        if annotation is bool:
            return value.lower() in ("1", "true", "yes", "on")
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)

        return value
