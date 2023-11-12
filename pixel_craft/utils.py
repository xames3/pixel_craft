"""\
Pixel Craft's Utilities
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, November 10 2023
Last updated on: Friday, November 10 2023
"""

from __future__ import annotations

import os.path as p
import re
import typing as t
from urllib.parse import urlparse

import cv2
import numpy as np

_MatLike = cv2.mat_wrapper.Mat | np.ndarray[t.Any, np.dtype[np.generic]]


def convert_case(name: str) -> str:
    """Convert ``PascalCase`` class names to ``SnakeCase``.

    :param name: Name of the class in PascalCase.
    :returns: Name conformed to SnakeCase.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def validate_url(url: str) -> bool:
    """Validate the authenticity of the URL using RegEx."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
        r"(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url) is not None


def is_local(file: str) -> bool:
    """Check whether the passed string is a URL or local file."""
    parsed = urlparse(file)
    if parsed.scheme in ("file", ""):
        return p.exists(parsed.path)
    return False


class History:
    """A wrapper for accessing image states."""

    def __init__(self) -> None:
        """Initialize ``History`` with an empty dictionary."""
        self.history = {}

    def __getattr__(self, __name: str) -> _MatLike:
        """Provide attribute-style access to the class contents."""
        try:
            return self.history[__name]
        except KeyError:
            raise AttributeError

    def __setattr__(self, __name: str, __value: _MatLike) -> None:
        """Set attribute to the history object."""
        if __name == "history":
            super().__setattr__(__name, __value)
        else:
            self.history[__name] = __value

    __getitem__ = __getattr__
