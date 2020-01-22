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
"""Needs a doc-string.
"""

import sys
import argparse

from frankenstein import FrankFitter, FourierBesselFitter


def help():
    pass


def parse_parameters():
    return {}


def load_uvdata(params):
    pass


def perform_fit(params):
    pass


def output_results(params):
    pass


def main():
    # Maybe some ASCII art?
    print("The world was to me a secret which I desired to devine.\n"
          "     - Mary Shelley, Frankenstein\n")

    params = parse_parameters()

    uv = load_uvdata(params)

    perform_fit(params)

    output_results(params)

    print("\n\n\nIT'S ALIVE\n\n")


if __name__ == "__main__":
    main()
