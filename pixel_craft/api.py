"""\
Pixel Craft's Main APIs
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, November 10 2023
Last updated on: Saturday, November 11 2023
"""

from __future__ import annotations

import typing as t

import cv2
import numpy as np
import requests

from .exceptions import ImageReadingError
from .utils import History
from .utils import convert_case
from .utils import is_local
from .utils import validate_url

if t.TYPE_CHECKING:
    from .utils import _MatLike

_Image = type["Image"]
_SupportedOperations = dict[str, _Image]


class ImageMeta(type):
    """Overrides the ``__getattribute__`` method for class attributes.

    This metaclass overrides the standard ``__getattribute__`` to allow
    easy access to the latest image from the history pool. Rest of the
    implementation stays the same.
    """

    def __getattribute__(cls, __name: str) -> t.Any | _MatLike:
        """Custom attribute access method favoring ``image`` call."""
        if __name == "image":
            return cls.history.image
        return super().__getattribute__(__name)


class Image(metaclass=ImageMeta):
    """Primary image class for loading, saving and displaying the image."""

    __supported_operations: _SupportedOperations = {}

    def __init_subclass__(cls) -> None:
        """Register the image processing operations.

        This method is called when a class is subclassed. This allows
        the ``Image`` class to keep a track of the derived classes.
        """
        cls.__supported_operations[convert_case(cls.__name__)] = cls

    def __new__(cls, path: str) -> _Image:
        """Create and return new ``Image`` object.

        :param path: Name of file to be loaded.
        :returns: Image object.
        """
        for operation, instance in cls.__supported_operations.items():
            setattr(cls, operation, instance.__call__)
        cls.history = History()
        if isinstance(path, str):
            if is_local(path):
                cls.filename, cls.url = path, None
                cls.history.image = cls.__read()
            else:
                cls.filename, cls.url = None, path
                cls.history.image = cls.__parse()
        else:
            raise TypeError("Expecting path (local or URL) to the image")
        cls.history.original = cls.history.image
        return cls

    @classmethod
    def __read(cls) -> _MatLike:
        """Loads an image from a file.

        This method loads an image from the specified file path and
        returns it. Currently, the following file formats are supported:

            - Windows bitmaps - *.bmp, *.dib
            - JPEG files - *.jpeg, *.jpg, *.jpe
            - JPEG 2000 files - *.jp2
            - Portable Network Graphics - *.png
            - WebP - *.webp
            - AVIF - *.avif
            - Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm
            - PFM files - *.pfm
            - Sun rasters - *.sr, *.ras
            - TIFF files - *.tiff, *.tif
            - OpenEXR Image files - *.exr
            - Radiance HDR - *.hdr, *.pic

        .. note::

            [1] The method determines the type of an image by the
                content, not by the file extension.
            [2] In the case of color images, the decoded images will
                have the channels stored in BGR order and NOT RGB.
        """
        return cv2.imread(cls.filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)

    @classmethod
    def __parse(cls) -> _MatLike:
        """Load an image from a URL.

        This method reads an image from a buffer in memory. If the
        buffer is too short or contains invalid data, the function
        returns an empty matrix.

        .. note::

            [1] In the case of color images, the decoded images will
                have the channels stored in BGR order and NOT RGB.
        """
        if validate_url(cls.url):
            response = requests.get(cls.url)
            buffer = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        else:
            raise ImageReadingError("Invalid image address")

    @classmethod
    def show(cls, **kwargs: t.Any) -> None:
        """Displays image in specified window, if any.

        :param title: Name of the window, defaults to ``Output Window``.
        :param size: Tuple of window width and height to be resized to,
                     defaults to ``None``.

        .. note::

            The specified window size is for the image area. Toolbars are
            NOT counted.
        """
        title = kwargs.pop("title", "Output Window")
        size = kwargs.get("size")
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cls.image)
        if size:
            cv2.resizeWindow(title, *size)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    @classmethod
    def save(cls, filename: str) -> None:
        """Saves the current image in history to a specified file.

        :param filename: Path where the image should be saved.

        .. seealso::

            [1] The image format is chosen based on the filename
                extension. Read the ``Image.__read()`` for supported file
                formats.
        """
        cv2.imwrite(filename, cls.image)

    @classmethod
    def compare(cls, *args: str, **kwargs: t.Any) -> None:
        """Display images side by side for visual comparison.

        :param title: Name of the window.
        """
        combined = []
        for arg in args:
            image = cls.history[arg]
            if len(image.shape) < 3:
                image = np.stack((image,) * 3, axis=-1)
            combined.append(image)
        cls.history.image = cv2.hconcat(combined)
        title = kwargs.pop("title", " v/s ".join(map(str.title, args)))
        cls.show(title=title, **kwargs)


class Grayscale(Image):
    """Class for converting an image to grayscale color space.

    The class converts the source image from its current color space to
    grayscale. In case of a transformation to-from RGB color space, the
    order of the channels should be specified explicitly (RGB or BGR).

    .. warning::

        If you use 8-bit images, the conversion will have some of the
        information lost.
    """

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Convert source image to grayscale."""
        cls.history.image = cv2.cvtColor(cls.image, 6)
        cls.history.grayscale = cls.history.image
        return cls


class Invert(Image):
    """Class for inverting the image."""

    @classmethod
    def __call__(cls, **kwargs) -> type[t.Self]:
        """Invert source image to a digital negative."""
        factor = kwargs.pop("factor", 255)
        cls.history.image = factor - cls.image
        cls.history.invert = cls.history.image
        return cls


class StandardBlur(Image):
    """Class for blurring image using normalized box filter."""

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Blurs the source image using the normalized box filter.

        :param kernel: Tuple of blurring kernel size, defaults
                       to ``(10, 10)``.
        :param anchor: Tuple of point value, defaults to ``(-1, -1)``.
                       This means the anchor is at the kernel's center.
        """
        cls.history.image = cv2.blur(
            cls.image,
            ksize=kwargs.pop("kernel", (10, 10)),
            anchor=kwargs.pop("anchor", (-1, -1)),
        )
        cls.history.standard_blur = cls.history.image
        return cls


class GaussianBlur(Image):
    """Class for blurring image using Gaussian filter."""

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Blurs the source image using a Gaussian filter.

        The method convolves the source image with the specified Gaussian
        kernel.

        :param kernel: Tuple of Gaussian kernel size. The kernel's width
                       and height can differ but they both must be
                       positive and odd.
        """
        cls.history.image = cv2.blur(cls.image, ksize=kwargs.pop("kernel"))
        cls.history.gaussian_blur = cls.history.image
        return cls


class Canny(Image):
    """Class for detecting edges using Canny Edge Detection."""

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Finds edges in an image using the Canny algorithm.

        The method finds edges in the source image and marks them in the
        output map edges using the Canny algorithm. The smallest value
        between ``threshold_1`` and ``threshold_2`` is used for edge
        linking.

        The largest value is used to find initial segments of strong
        edges.

        :param threshold_1: First threshold for the hysteresis process.
        :param threshold_2: Second threshold for the hysteresis process.
        :param aperture_size: Aperture size for the Sobel operator,
                              defaults to ``3``.
        :param accurate: A boolean flag, indicating whether a more
                         accurate L2 norm should be used to calculate
                         the image gradient magnitude, or whether the
                         default L1 norm is enough for the operation,
                         defaults to ``False``.
        :returns: An output edge map, which has the same size as image.

        .. seealso::
            [1] http://en.wikipedia.org/wiki/Canny_edge_detector
        """
        cls.history.image = cv2.Canny(
            cls.image,
            threshold1=kwargs.pop("threshold_1", 20),
            threshold2=kwargs.pop("threshold_2", 90),
            apertureSize=kwargs.pop("aperture_size", 3),
            L2gradient=kwargs.pop("accurate", False),
        )
        cls.history.canny = cls.history.image
        return cls
