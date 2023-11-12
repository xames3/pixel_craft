"""\
Pixel Craft's Command Line API
==============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, October 16 2023
Last updated on: Friday, November 10 2023

This module hosts the main argument parser object which allows user to
interact with Pixel Craft's APIs over the command line.

Usage Example::
---------------

    .. code-block:: console

        $ python3 -c "import pixel_craft; pixel_craft.cli.main()" --help

"""

from __future__ import annotations

from . import __version__ as version


def main() -> int:
    """Primary application entrpoint.

    This function is called at the entrypoint, meaning that when the
    user runs this function, it will display the command line interface
    for Pixel Craft.

    Run as standalone python application.
    """
    return 0
