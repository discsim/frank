Generating a UVTable from an MSTable for input to ``frank``
===========================================================

As noted in the `Quickstart <../quickstart.rst>`_, frank takes as input a UVTable with
columns of `u [\lambda]     v [\lambda]      Re(V) [Jy]     Im(V) [Jy]     Weight [Jy^-2]`.

A UVTable is just the ASCII form of an MSTable (measurement set), which is the default data format internal to CASA.
In many cases we must thus convert the MSTable internally in CASA to a UVTable.
To do this, the ``uvplot`` package is a good option.
You can install it inside CASA with `casa-pip install uvplot` (see the `uvplot repo <https://github.com/mtazzari/uvplot#installation>`_).
Then use it to extract a UVTable by following `these simple steps <https://github.com/mtazzari/uvplot#2-exporting-visibilities-from-ms-table-to-uvtable-ascii>`_.

As noted there, we recommend performing a CASA `split` command with `keepflags=False` before exporting the UVTable,
as this ensures that only valid visibilities are exported.

In tests we've found that channel averaging an MSTable tends to have a weak effect on the frank fit,
though it's worth performing a fit on your dataset without any averaging to verify.
The increased density of (u, v) points under no channel averaging may in some cases improve the data's SNR
at long baselines enough to increase the maximum baseline out to which frank fits.
