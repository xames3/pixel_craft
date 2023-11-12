"""\
Pixel Craft's Exceptions
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, November 10 2023
Last updated on: Friday, November 10 2023
"""

import logging
import typing as t

_logger = logging.getLogger("pixel_craft.main")


class PixelCraftException(Exception):
    """Base class for capturing all the exceptions raised by the Pixel
    Craft module.

    This exception class serves as the primary entrypoint for capturing
    and logging exceptions related to all the image processing
    operations.

    :var _description: A human-readable description or message
                       explaining the reason for the exception.
    """

    _description: str

    def __init__(self, description: str | None, *args: t.Any) -> None:
        if description:
            self._description = description
        _logger.error(self._description)
        super().__init__(self._description, *args)


class ImageReadingError(PixelCraftException):
    """Raise this error when the image file cannot be loaded to open it
    using OpenCV.
    """
