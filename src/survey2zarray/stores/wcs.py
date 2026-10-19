"""Rebuild a WCS from the numbers a store keeps beside its pixels.

Both store types record the same four parameters per image or cutout, so both
reconstruct the same way."""

from collections.abc import Sequence

import numpy as np
from astropy.wcs import WCS


def reconstruct_wcs(
    crval: np.ndarray | Sequence[float],
    crpix: np.ndarray | Sequence[float],
    cd: np.ndarray | Sequence[Sequence[float]],
    ctype: str,
    radesys: str | None = None,
    equinox: float | None = None,
) -> WCS:
    """Rebuild a WCS from stored numeric parameters.

    Rebuilding from numbers avoids reparsing the header string, which is far
    slower. Only ``cd`` is set, since setting ``cdelt`` or ``pc`` as well would
    make the transformation ambiguous.

    Parameters
    ----------
    crval
        World coordinates at the reference pixel.
    crpix
        Reference pixel.
    cd
        Linear transformation matrix.
    ctype
        Axis types joined by ``'|'``.
    radesys
        Celestial reference frame, such as ``'ICRS'`` or ``'FK5'``.
    equinox
        Equinox year for frames which use one.

    Returns
    -------
    result
        WCS with as many axes as ``crval`` has entries.
    """
    naxis = len(crval)
    wcs = WCS(naxis=naxis)
    wcs.wcs.crpix = crpix
    wcs.wcs.crval = crval
    wcs.wcs.cd = cd
    wcs.wcs.ctype = list(str(ctype).split("|"))
    if radesys:
        wcs.wcs.radesys = str(radesys)
    if equinox is not None and np.isfinite(equinox):
        wcs.wcs.equinox = float(equinox)
    return wcs
