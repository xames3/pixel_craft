"""\
Pixel Craft's Main APIs
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, November 10 2023
Last updated on: Saturday, November 11 2023
"""

from __future__ import annotations

import typing as t
import warnings

import cv2
import matplotlib.colors as mcolors
import numpy as np
import requests
from matplotlib import pyplot as plt

from .exceptions import ImageReadingError
from .utils import History
from .utils import convert_case
from .utils import is_local
from .utils import validate_url

if t.TYPE_CHECKING:
    from .utils import _MatLike

_Image = type["Image"]
_SupportedOperations = dict[str, _Image]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 200

axes = plt.gca()
axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)

b = mcolors.TABLEAU_COLORS["tab:blue"]
g = mcolors.TABLEAU_COLORS["tab:green"]
r = mcolors.TABLEAU_COLORS["tab:red"]
n = mcolors.TABLEAU_COLORS["tab:gray"]


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

    _help: str
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
        size = kwargs.pop("size", None)
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
        if "size" in kwargs:
            warnings.warn("Using size while comparing isn't recommended")
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

    _help: str = (
        "Convert the source image from its current color space to grayscale. "
        "In case of a transformation to-from RGB color space, the order of "
        "the channels should be specified explicitly (RGB or BGR)."
    )

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Convert source image to grayscale."""
        cls.history.image = cv2.cvtColor(cls.image, 6)
        cls.history.grayscale = cls.history.image
        return cls


class Invert(Image):
    """Class for inverting the image."""

    _help: str = "Create a digital negative of the source image."

    @classmethod
    def __call__(cls, **kwargs) -> type[t.Self]:
        """Invert source image to a digital negative."""
        factor = kwargs.pop("factor", 255)
        cls.history.image = factor - cls.image
        cls.history.invert = cls.history.image
        return cls


class StandardBlur(Image):
    """Class for blurring image using normalized box filter."""

    _help: str = "Blur the source image using normalized box filter."

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

    _help: str = "Blur the source image using a Gaussian filter."

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Blurs the source image using a Gaussian filter.

        The method convolves the source image with the specified Gaussian
        kernel.

        :param kernel: Tuple of Gaussian kernel size. The kernel's width
                       and height can differ but they both must be
                       positive and odd.
        :param anchor: Tuple of point value, defaults to ``(-1, -1)``.
                       This means the anchor is at the kernel's center.
        """
        cls.history.image = cv2.blur(
            cls.image,
            ksize=kwargs.pop("kernel"),
            anchor=kwargs.pop("anchor", (-1, -1)),
        )
        cls.history.gaussian_blur = cls.history.image
        return cls


class Canny(Image):
    """Class for detecting edges using Canny Edge Detection."""

    _help: str = "Finds edges in an image using the Canny algorithm."

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


class Sobel(Image):
    """Class for detecting edges using Sobel operator/filter."""

    _help: str = "Finds edges in an image using Sobel operator."

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Finds edges in an image using Sobel operator.

        The Sobel operator, sometimes called the ``Sobel-Feldman``
        operator or Sobel filter, is used in image processing and
        computer vision, particularly within edge detection algorithms
        where it creates an image emphasising edges.

        The operator uses two 3x3 kernels which are convolved with the
        original image to calculate approximations of the derivatives -
        one for horizontal changes, and one for vertical. More formally,
        speaking, since the intensity function of a digital image is
        only known at discrete points, derivatives of this function
        cannot be defined unless we assume that there is an underlying
        differentiable intensity function that has been sampled at the
        image points.

        With some additional assumptions, the derivative of the
        continuous intensity function can be computed as a function on
        the sampled intensity function, i.e. the digital image. It turns
        out that the derivatives at any particular point are functions
        of the intensity values at virtually all image points.

        However, approximations of these derivative functions can be
        defined at lesser or larger degrees of accuracy. The Sobel
        Operator is a discrete differentiation operator. It computes an
        approximation of the gradient of an image intensity function.

        :param image_depth: Output image depth, defaults to ``-1``.
        :param kernel_h: Horizontal kernel numpy array, defaults to
                         ``[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]``.
        :param kernel_v: Vertical kernel numpy array, defaults to
                         ``[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]``.
        :param anchor: Tuple of point value, defaults to ``(-1, -1)``.
                       This means the anchor is at the kernel's center.
        :param border_type: Pixel extrapolation method, defaults to
                            ``4``.

        .. seealso::
            [1] https://en.wikipedia.org/wiki/Sobel_operator
            [2] https://shorturl.at/glnEM
            [3] https://shorturl.at/jkuG0
        """
        cls.history.image = cv2.filter2D(
            cls.image,
            ddepth=kwargs.pop("image_depth", -1),
            kernel=(
                kwargs.pop(
                    "kernel_h", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                )
                + kwargs.pop(
                    "kernel_v", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                )
            ),
            anchor=kwargs.pop("anchor", (-1, -1)),
            borderType=kwargs.pop("border_type", 4),
        )
        cls.history.sobel = cls.history.image
        return cls


class Prewitt(Image):
    """Class for detecting edges using Prewitt operator/filter."""

    _help: str = "Finds edges in an image using Prewitt operator."

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Finds edges in an image using Prewitt operator.

        The Prewitt operator is used in image processing, particularly
        within edge detection algorithms. Technically, it is a discrete
        differentiation operator, computing an approximation of the
        gradient of the image intensity function. At each point in the
        image, the result of the Prewitt operator is either the
        corresponding gradient vector or the norm of this vector.

        The Prewitt operator is based on convolving the image with a
        small, separable, and integer valued filter in horizontal and
        vertical directions and is therefore relatively inexpensive in
        terms of computations like Sobel.

        :param image_depth: Output image depth, defaults to ``-1``.
        :param kernel_h: Horizontal kernel numpy array, defaults to
                         ``[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]``.
        :param kernel_v: Vertical kernel numpy array, defaults to
                         ``[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]``.
        :param anchor: Tuple of point value, defaults to ``(-1, -1)``.
                       This means the anchor is at the kernel's center.
        :param border_type: Pixel extrapolation method, defaults to
                            ``4``.

        .. seealso::
            [1] https://en.wikipedia.org/wiki/Prewitt_operator
        """
        cls.history.image = cv2.filter2D(
            cls.image,
            ddepth=kwargs.pop("image_depth", -1),
            kernel=(
                kwargs.pop(
                    "kernel_h", np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                )
                + kwargs.pop(
                    "kernel_v", np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                )
            ),
            anchor=kwargs.pop("anchor", (-1, -1)),
            borderType=kwargs.pop("border_type", 4),
        )
        cls.history.prewitt = cls.history.image
        return cls


class Histogram(Image):
    """Class for representing image as Histogram."""

    _help: str = "Represent image as histogram."

    @classmethod
    def __call__(cls, **kwargs: t.Any) -> type[Image]:
        """Show histogram plot using Matplotlib."""
        channels = (n,) if len(cls.image.shape) < 3 else (b, g, r)
        for idx, channel in enumerate(channels):
            x = np.arange(kwargs.pop("size", 256))
            y = cv2.calcHist(
                [cls.image],
                channels=[idx],
                mask=kwargs.pop("mask", None),
                histSize=[kwargs.pop("size", 256)],
                ranges=kwargs.pop("ranges", [0, 256]),
            ).flatten()
            plt.plot(x, y, alpha=kwargs.pop("alpha", 0.7), color=channel)
            plt.fill_between(
                x, y, alpha=kwargs.pop("fill", 0.3), color=channel
            )
        plt.xlabel(kwargs.pop("xlabel", "Levels"), fontname="Helvetica")
        plt.ylabel(kwargs.pop("ylabel", "Frequency"), fontname="Helvetica")
        plt.show()
        return cls


path = "/Users/akshay/Developer/masters/4_quarter/csc_481/2/Lena.png"

# image = Image(path)
# image.grayscale().prewitt().show()
