from collections import namedtuple

"""
Container for results of the background model fit.

R. Ryan
Jun 2026

Attributes
----------
alpha : float
    Best-fit scaling factor applied to the background model.
chi2 : float
    Chi-squared value of the fit.
npix : int
    Number of pixels used in the fit.
chi2nu : float
    Reduced chi-squared (chi2 / (npix - 1)).
resid : ndarray
    Residual image (data - model) / uncertainty.
"""

ModelResult = namedtuple('ModelResult', ('alpha', 'chi2', 'npix', 'chi2nu', 'resid'))
