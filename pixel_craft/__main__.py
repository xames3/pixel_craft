"""\
Pixel Craft's Entry Point API
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, October 16 2023
Last updated on: Friday, November 10 2023

This module calls the ``pixel_craft.cli.main()`` to act as an entrypoint
for the main porject. The function returns an exit code of ``0`` if the
commands are executed successfully else it'll return 1 and an error log
will be displayed.

For more information about how the command line interface works, please
see ``pixel_craft.cli.py``.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
