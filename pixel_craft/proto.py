"""\
CSC 481 Introduction to Image Processing
========================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, October 12 2023
Last updated on: Saturday, October 14 2023
"""

from __future__ import annotations

import logging
import os.path as p
import re
import typing as t

import cv2
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

__author__ = "Akshay Mestry <xa@mes3.dev>"
__version__ = "0.0.8"

MatLike = cv2.mat_wrapper.Mat | np.ndarray[t.Any, np.dtype[np.generic]]
MatShape = t.Sequence[int]
Size = t.Sequence[int]

log = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="%(msg)s")

plt.style.use("fast")
plt_axis = plt.gca()
plt_axis.spines["top"].set_visible(False)
plt_axis.spines["right"].set_visible(False)

b = mcolors.TABLEAU_COLORS["tab:blue"]
g = mcolors.TABLEAU_COLORS["tab:green"]
r = mcolors.TABLEAU_COLORS["tab:red"]
n = mcolors.TABLEAU_COLORS["tab:gray"]
a = np.round(np.clip(np.random.rand(), 0, 1), 1)


class ImageException(Exception):
    """Base class for capturing all the exceptions raised by the image
    object.

    This exception class serves as the primary entrypoint for capturing
    and logging exceptions related to image processing operations.

    :var _description: A human-readable description or message
                       explaining the reason for the exception.
    """

    _description: str

    def __init__(self, description: str | None = None) -> None:
        """Initialize ``ImageException`` with error description."""
        if description:
            self._description = description
        log.error(self._description)
        super().__init__(self._description)


class UnreadableImageError(ImageException):
    """Error to be raised when the object is neither a filepath or an
    instance of image object.

    .. versionadded:: 0.0.2
    """


class ImageNotFoundError(ImageException):
    """Error to be raised when the desired image couldn't be found."""


def dynamic_property(method: type[Image]) -> t.Callable[[t.Any], t.Any]:
    """Generate dynamic classmethod-property from a callback function.

    This function is used to create a classmethod-property that returns
    an instance of the provided callback class. The resulting property
    can be accessed on the class without the need to instantiate the
    callback class, making it a powerful tool for accessing shared
    resources or services.

    The ``@classmethod`` decorator ensures that the ``wrapper`` property
    can be accessed on the class itself, rather than on an instance of
    the class. The ``@property`` decorator makes it possible to access
    the ``wrapper`` method as if it were a regular attribute or method,
    even though it's a classmethod-property.

    :param callback: Operation method to execute as a callback.

    .. versionadded:: 0.0.6
        Added support for generating dynamic properties from a callback
        function.
    """

    @classmethod  # type:ignore[misc]
    @property
    def wrapper(cls):
        return method

    return wrapper


class Image:
    """Primary image class for loading, saving and displaying the image.

    This class acts as primary entrypoint for all the image processing
    related task. It allows loading, saving and showing the loaded
    image object as part of its primary operartions. Furthermore, the
    class can be extended for even more complex operations by inheriting
    it and adding necessary functionalities.

    :param filepath_or_image: Absolute file path to the image file or the
                             numpy array image data loaded from a file.
    :var _operations: Dictionary of all the supported operations by
                      the ``Image`` class.
    :var _methods: Dictionary of all the supported methods by the
                   subclasses of the ``Image`` class.
    :var _options: Mutable sharable instance of the image object.

    :raises UnreadableImageError: If the passed argument could not be
                                  resolved as an image object or a file
                                  path.
    :raises ImageNotFoundError: If the class is not able to find an image
                                file to read through the provided path.
    """

    _operations: dict[str, type[Image]] = {}
    _methods: dict[str, type[Image]] = {}
    _options: dict[str, t.Any] = {
        "image": np.zeros((1, 1)),
        "images": {},
        "is_grayscale": False,
    }

    def __init_subclass__(cls) -> None:
        """Register all the derived operations with their names.

        .. versionadded:: 0.0.3
            Added supporting for registering the derived classes.
        """
        klass = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        cls._operations[klass] = cls

    def __new__(cls, filepath_or_image: str) -> Image:
        """Initialize ``Image`` with image object."""
        for _, klass in cls._operations.items():
            for method, _klass in klass._methods.items():
                setattr(cls, method, dynamic_property(_klass))
        if isinstance(filepath_or_image, str):
            cls.filepath = filepath_or_image
            cls._options["image"] = cls._read()
        elif isinstance(filepath_or_image, np.ndarray):
            cls.filepath = None
            cls._options["image"] = filepath_or_image
        else:
            raise UnreadableImageError(
                f"Object {filepath_or_image!r} could not be resolved as"
                " an image or a filepath"
            )
        cls._options["images"]["original"] = cls._options["image"]
        return cls  # type: ignore[return-value]

    @classmethod
    def _read(cls) -> MatLike:
        """Read input filepath as image object."""
        if not p.exists(cls.filepath):
            raise ImageNotFoundError(
                f"Unable to find image file on path {cls.filepath!r}"
            )
        return cv2.imread(cls.filepath).astype(np.uint8)

    @classmethod
    def show(cls, **kwargs: t.Any) -> None:
        """Display the image as a named window.

        This method leverages and displays the loaded image using
        OpenCV's ``imshow`` method and waits until a key (Esc) is
        pressed to close the displayed window.
        """
        if cls._options["image"] is None:
            raise ImageNotFoundError("No image found to display")
        title = kwargs.pop("title", "Output")
        size = kwargs.pop("size", (200, 200))
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cls._options["image"])
        cv2.resizeWindow(title, *size)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @classmethod
    def compare(cls, *args: t.Sequence[str], **kwargs: t.Any) -> None:
        """Display original image with transformed image side by side
        for visual comparison.
        """
        if not cls._options["images"]:
            raise ImageNotFoundError("No images found to display")
        cmb = cv2.hconcat([cls._options["images"][image] for image in args])
        title = kwargs.pop("title", " v/s ".join(map(str.title, args)))
        size = kwargs.pop("size", (200 * len(args), 200))
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cmb)
        cv2.resizeWindow(title, *size)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @classmethod
    def save(cls, output: str) -> None:
        """Save the image to a specified output filepath using OpenCV's
        ``imwrite`` method.

        :param output: Path where the image should be saved.
        """
        if cls._options["image"] is None:
            raise ImageNotFoundError("No image found to save")
        cv2.imwrite(output, cls._options["image"])
        log.info(f"Image saved to {output!r}")


class Process(Image):
    """Base class for registering image processes.

    .. versionadded:: 0.0.4
        Added supporting for registering the derived processes.
    """

    def __init_subclass__(cls) -> None:
        """Initialize ``Process`` with image object."""
        klass = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        cls._methods[klass] = cls

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Implement filter applying method."""
        raise NotImplementedError


class Represent(Image):
    """Base class for registering image visualizing techniques.

    .. versionadded:: 0.0.7
        Added supporting for registering the derived techniques.
    """

    def __init_subclass__(cls) -> None:
        """Initialize ``Represent`` with image object."""
        klass = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        cls._methods[klass] = cls

    @classmethod
    def show(cls, **kwargs) -> None:
        """Implement image representing method."""
        raise NotImplementedError


class RawFilter(Process):
    """Class for applying raw image filter.

    ..versionadded:: 0.0.8
        Added support for raw image filtering.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Apply filter on the input image."""
        kernel = kwargs.pop(
            "kernel", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        )
        cls._options["is_grayscale"] = True
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        ih, iw = cls._options["image"].shape
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        zeros = np.zeros_like(cls._options["image"])
        for idx in range(ih):
            for jdx in range(iw):
                pixels = 0
                for mdx in range(kh):
                    for ndx in range(kw):
                        p_idx = idx - ph + mdx
                        p_jdx = jdx - pw + ndx
                        if (
                            p_idx < 0
                            or p_idx >= ih
                            or p_jdx < 0
                            or p_jdx >= iw
                        ):
                            tmp = 0
                        else:
                            tmp = cls._options["image"][p_idx, p_jdx]
                        pixels += tmp * kernel[mdx, ndx]
                zeros[idx, jdx] = pixels
        cls._options["image"] = zeros
        cls._options["image"] = cv2.resize(cls._options["image"], (ih, iw))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["raw filter"] = cls._options["image"]
        return cls


class SobelFilter(Process):
    """Class for applying sobel image filter.

    .. versionadded:: 0.0.8
        Added support for Sobel image filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Apply sobel filter to the input image."""
        cls._options["is_grayscale"] = True
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.filter2D(
            cls._options["image"], -1, (kernel_x + kernel_y)
        )
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["sobel filter"] = cls._options["image"]
        return cls


class PrewittFilter(Process):
    """Class for applying prewitt image filter.

    .. versionadded:: 0.0.8
        Added support for Prewitt image filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Apply prewitt filter to the input image."""
        cls._options["is_grayscale"] = True
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.filter2D(
            cls._options["image"], -1, (kernel_x + kernel_y)
        )
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["prewitt filter"] = cls._options["image"]
        return cls


class BlurringFilter(Process):
    """Class for applying blurring image filter.

    .. versionadded:: 0.0.8
        Added support for Blurring image filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Apply blurring filter to the input image."""
        cls._options["is_grayscale"] = True
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.filter2D(cls._options["image"], -1, kernel)
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["blurring filter"] = cls._options["image"]
        return cls


class PointFilter(Process):
    """Class for applying point image filter.

    .. versionadded:: 0.0.8
        Added support for Point image filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Apply point filter to the input image."""
        cls._options["is_grayscale"] = True
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.filter2D(cls._options["image"], -1, kernel)
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["point filter"] = cls._options["image"]
        return cls


class ContrastStretching(Process):
    """Class for performing contrast stretching.

    .. versionadded:: 0.0.8
        Added support for contrast stretching.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Perform contrast stretching."""
        cls._options["is_grayscale"] = True
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cmin = np.min(cls._options["image"])
        cmax = np.max(cls._options["image"])
        cls._options["image"] = np.uint8(
            (cls._options["image"] - cmin) * ((255 - 0) / (cmax - cmin)) + 0
        )
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["contrast stretch"] = cls._options["image"]
        return cls


class Grayscale(Process):
    """Class for converting images to grayscale."""

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Convert original image to grayscale."""
        cls._options["is_grayscale"] = True
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["grayscale"] = cls._options["image"]
        return cls


class Invert(Process):
    """Class for applying inverting filter on the image.

    .. versionadded:: 0.0.5
        Added supporting for invert filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Invert original image to generate a digital negative."""
        factor = kwargs.pop("factor", 255)
        cls._options["image"] = factor - cls._options["image"]
        cls._options["images"]["invert"] = cls._options["image"]
        return cls


class Blur(Process):
    """Class for applying blur on the image.

    .. versionadded:: 0.0.7
        Added supporting for blur filter.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Blur original image."""
        kernel = kwargs.pop("kernel", (10, 10))
        cls._options["image"] = cv2.blur(cls._options["image"], kernel)
        cls._options["images"]["blur"] = cls._options["image"]
        return cls


class ImproveContrast(Process):
    """Class for performing histogram equalization of the image.

    .. versionadded:: 0.0.7
        Added support for improving the contrast of the image using
        histogram equalization.
    """

    @classmethod
    def apply(cls, **kwargs) -> type[t.Self]:
        """Improve contrast of the original image."""
        cls._options["image"] = cls._options["images"]["original"]
        y, x, _ = cls._options["image"].shape
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 6)
        cls._options["image"] = cv2.equalizeHist(cls._options["image"])
        cls._options["image"] = cv2.resize(cls._options["image"], (y, x))
        cls._options["image"] = cv2.cvtColor(cls._options["image"], 8)
        cls._options["images"]["histogram equalized"] = cls._options["image"]
        return cls


class Histogram(Represent):
    """Class for representing image as histogram.

    .. versionadded:: 0.0.7
        Added support for visualizing histograms.
    """

    @classmethod
    def show(cls, **kwargs) -> None:
        """Show histogram plot using Matplotlib."""
        channels = (n,) if cls._options["is_grayscale"] else (b, g, r)
        image = kwargs.pop("image", None)
        if image is None:
            image = cls._options["image"]
        else:
            image = cls._options["images"][image]
        ranges = kwargs.pop("ranges", [0, 256])
        size = kwargs.pop("size", 256)
        mask = kwargs.pop("mask", None)
        for idx, channel in enumerate(channels):
            histogram = cv2.calcHist(
                [image],
                channels=[idx],
                mask=mask,
                histSize=[size],
                ranges=ranges,
            ).flatten()
            x_axis = np.arange(size)
            plt.plot(x_axis, histogram, alpha=a, color=channel)
            plt.fill_between(x_axis, 0, histogram, alpha=0.3, color=channel)
        plt.xlabel("Levels", fontname="Helvetica")
        plt.ylabel("Frequency", fontname="Helvetica")
        plt.show()


path = "/Users/akshay/Developer/masters/4_quarter/csc_481/2/Lena.png"

image = Image(path)
# image.grayscale.apply().improve_contrast.apply().compare("original", "grayscale", "histogram equalized")
# image.grayscale.apply().improve_contrast.apply().histogram.show(image="histogram equalized")
image.grayscale.apply().contrast_stretching.apply().compare("grayscale", "contrast stretch")
# image.grayscale.apply().sobel_filter.apply().prewitt_filter.apply().point_filter.apply().blurring_filter.apply().compare(
#     "grayscale",
#     "sobel filter",
#     "prewitt filter",
#     "point filter",
#     "blurring filter",
# )
# image.grayscale.apply().blur.apply().compare("original", "grayscale", "blur")
# image.grayscale.apply().show()
