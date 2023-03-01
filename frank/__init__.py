# Frankenstein: 1D disc brightness profile reconstruction from Fourier data
# using non-parametric Gaussian Processes
#
# Copyright (C) 2019-2020  R. Booth, J. Jennings, M. Tazzari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
__version__ = "1.2.1"

from frank import constants
from frank import geometry
from frank import hankel
from frank import io
from frank import radial_fitters
from frank import debris_fitters
from frank import utilities

def enable_logging(log_file=None):
    """Turn on internal logging for Frankenstein

    Parameters
    ----------
    log_file : string, optional
        Output filename to which logging messages are written.
        If not provided, logs will only be printed to the screen
    """
    import logging

    if log_file is not None:
        handlers = [ logging.FileHandler(log_file, mode='w'),
                     logging.StreamHandler()
                     ]
    else:
        handlers = [ logging.StreamHandler() ]

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=handlers
                        )
