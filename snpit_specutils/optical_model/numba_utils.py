import numpy as np
import numba as nb

"""
These functions are the primary workhorse of the code, and so are accelerated
using numba.

Written by R. Ryan
Jun 2026
"""

@nb.njit
def horner1(c, z):
    """
    Evaluate a 1D polynomial at z using Horner's method.

    Parameters
    ----------
    c : ndarray, shape (n,)
        Polynomial coefficients ordered from lowest to highest degree.
    z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at z.
    """
    nz = c.shape[0]
    p = c[-1]
    for i in range(nz-2, -1, -1):
        p *= z
        p += c[i]
    return p


@nb.njit
def horner2(c, y, z):
    """
    Evaluate a 2D polynomial at (y, z) using nested Horner's method.

    Parameters
    ----------
    c : ndarray, shape (ny, nz)
        Polynomial coefficients, where c[i, j] corresponds to y^i z^j.
    y, z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at (y, z).
    """
    ny = c.shape[0]
    p = horner1(c[-1], z)
    for i in range(ny-2, -1, -1):
        p *= y
        p += horner1(c[i], z)
    return p


@nb.njit
def horner3(c, x, y, z):
    """
    Evaluate a 3D polynomial at (x, y, z) using nested Horner's method.

    Parameters
    ----------
    c : ndarray, shape (nx, ny, nz)
        Polynomial coefficients, where c[i, j, k] corresponds to x^i y^j z^k.
    x, y, z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at (x, y, z).
    """
    nx = c.shape[0]
    p = horner2(c[-1], y, z)
    for i in range(nx-2, -1, -1):
        p *= x
        p += horner2(c[i], y, z)
    return p


@nb.njit(parallel=True)
def disperse_pairwise(Xij, Yij, Cijk, Dijk, x, y, w):
    """
    Compute dispersed coordinates for pairwise inputs using Numba.

    This version assumes x, y, and w have identical shapes and computes
    one output coordinate pair per input triplet.

    Parameters
    ----------
    Xij, Yij : ndarray, shape (nx, ny)
        Polynomial coefficients mapping undispersed coordinates to
        Focal Plane Assembly (FPA) space.
    Cijk, Dijk : ndarray, shape (nw, nx, ny)
        Polynomial coefficients for dispersion in FPA space.
    x, y : ndarray
        Undispersed FPA coordinates.
    w : ndarray
        Transformed wavelength coordinate.

    Returns
    -------
    xt, yt : ndarray
        Dispersed FPA coordinates.
    """
    xt = np.empty_like(x)
    yt = np.empty_like(x)

    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]
        ww = w.flat[idx]

        dy = horner3(Cijk, ww, xx, yy)
        dx = horner3(Dijk, dy, xx, yy)

        xmpa = horner2(Xij, xx, yy)
        ympa = horner2(Yij, xx, yy)

        xt.flat[idx] = xmpa + dx
        yt.flat[idx] = ympa + dy
        
    return xt, yt


@nb.njit(parallel=True)
def disperse(Xij, Yij, Cijk, Dijk, x, y, w):
    """
    Compute dispersed coordinates on a grid using Numba.

    This version assumes x and y are 1D arrays of positions, and w is a
    1D array of wavelengths. Outputs are 2D arrays indexed by
    (pixel_index, wavelength_index).

    Parameters
    ----------
    Xij, Yij : ndarray, shape (nx, ny)
        Polynomial coefficients mapping undispersed coordinates to
        Focal Plane Assembly (FPA) space.
    Cijk, Dijk : ndarray, shape (nw, nx, ny)
        Polynomial coefficients for dispersion in FPA space.
    x, y : ndarray, shape (npix,)
        Undispersed FPA coordinates.
    w : ndarray, shape (nlam,)
        Transformed wavelength coordinate.

    Returns
    -------
    xt, yt : ndarray, shape (npix, nlam)
        Dispersed FPA coordinates.
    """
    npix = x.size
    nlam = w.size

    dim = (npix, nlam)
    xt = np.empty(dim, dtype=float)
    yt = np.empty(dim, dtype=float)

    for idx in nb.prange(npix):
        xx = x.flat[idx]
        yy = y.flat[idx]

        xmpa = horner2(Xij, xx, yy)
        ympa = horner2(Yij, xx, yy)

        for jdx in nb.prange(nlam):
            ww = w.flat[jdx]

            dy = horner3(Cijk, ww, xx, yy)
            dx = horner3(Dijk, dy, xx, yy)

            xt[idx, jdx] = xmpa + dx
            yt[idx, jdx] = ympa + dy
    return xt, yt


@nb.njit(parallel=True)
def deriv_pairwise(dCijk, Cijk, dDijk, x, y, w):
    """
    Compute derivatives of dispersed coordinates with respect to wavelength
    for pairwise inputs using Numba.

    Parameters
    ----------
    dCijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Cijk coefficients with respect to the transformed
        wavelength coordinate.
    Cijk : ndarray, shape (nw, nx, ny)
        Original dispersion coefficients.
    dDijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Dijk coefficients with respect to the dispersion
        coordinate.
    x, y : ndarray
        Undispersed FPA coordinates.
    w : ndarray
        Transformed wavelength coordinate.

    Returns
    -------
    dxdw, dydw : ndarray
        Derivatives of dispersed FPA coordinates with respect to transformed
        wavelength.
    """
    dxdw = np.empty_like(x)
    dydw = np.empty_like(x)
    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]
        ww = w.flat[idx]

        dy = horner3(Cijk, ww, xx, yy)

        dy_dw = horner3(dCijk, ww, xx, yy)
        dx_dy = horner3(dDijk, dy, xx, yy)
        
        dxdw[idx] = dy_dw * dx_dy
        dydw[idx] = dy_dw

    return dxdw, dydw    


@nb.njit(parallel=True)
def deriv(dCijk, Cijk, dDijk, x, y, w):
    """
    Compute derivatives of dispersed coordinates with respect to wavelength
    on a grid using Numba.

    Parameters
    ----------
    dCijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Cijk coefficients with respect to the transformed
        wavelength coordinate.
    Cijk : ndarray, shape (nw, nx, ny)
        Original dispersion coefficients.
    dDijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Dijk coefficients with respect to the dispersion
        coordinate.
    x, y : ndarray, shape (npix,)
        Undispersed FPA coordinates.
    w : ndarray, shape (nlam,)
        Transformed wavelength coordinate.

    Returns
    -------
    dxdw, dydw : ndarray, shape (npix, nlam)
        Derivatives of dispersed FPA coordinates with respect to transformed
        wavelength.
    """
    npix = x.size
    nlam = w.size

    dim = (npix, nlam)
    dxdw = np.empty(dim, dtype=float)
    dydw = np.empty(dim, dtype=float)

    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]

        for jdx in nb.prange(nlam):
            ww = w.flat[jdx]

            dy = horner3(Cijk, ww, xx, yy)
            
            dy_dw = horner3(dCijk, ww, xx, yy)
            dx_dy = horner3(dDijk, dy, xx, yy)
            
            dxdw[idx, jdx] = dy_dw * dx_dy
            dydw[idx, jdx] = dy_dw

    return dxdw, dydw    

