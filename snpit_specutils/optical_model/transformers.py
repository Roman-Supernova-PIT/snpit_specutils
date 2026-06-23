import numpy as np

"""
The optical model will be a polynomial of a variable `x`, which is
in term mapped to wavelength by one of these transformers

Written by R. Ryan
Jun 2026
"""



class LogTransformer:
    """
    Logarithmic wavelength transformer.

    Transforms physical wavelength values into a log-scaled coordinate
    relative to a reference wavelength, and provides inverse and derivative
    mappings.
    """
    ln10 = np.log(10.)

    def __init__(self, lam0):
        """
        Parameters
        ----------
        lam0 : float
            Reference wavelength.
        """
        self.lam0 = lam0

    def evaluate(self, lam):
        """
        Transform physical wavelength values into log-scaled coordinates.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        w : ndarray
            Log-scaled wavelength coordinates.
        """
        return np.log10(lam/self.lam0)

    def invert(self, w):
        """
        Invert the log-scaled wavelength transformation.

        Parameters
        ----------
        w : array_like
            Log-scaled wavelength coordinates.

        Returns
        -------
        lam : ndarray
            Physical wavelengths.
        """
        return self.lam0 * (10.0 ** w)

    def deriv(self, lam):
        """
        Compute dλ/dw for the log-wavelength transform.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        dldw : ndarray
            Derivative of wavelength with respect to the transformed coordinate.
        """
        return lam * self.ln10


class LinearTransformer:
    """
    Linear wavelength transformer.

    Transforms physical wavelength values into a linear coordinate
    relative to a reference wavelength, and provides inverse and derivative
    mappings.
    """

    def __init__(self, lam0):
        """
        Parameters
        ----------
        lam0 : float
            Reference wavelength.
        """
        self.lam0 = lam0

    def evaluate(self, lam):
        """
        Transform physical wavelength values into linear coordinates.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        w : ndarray
            Linear wavelength coordinates.
        """
        return lam - self.lam0

    def invert(self, w):
        """
        Invert the linear wavelength transformation.

        Parameters
        ----------
        w : array_like
            Linear wavelength coordinates.

        Returns
        -------
        lam : ndarray
            Physical wavelengths.
        """
        return w + self.lam0

    def deriv(self, lam):
        """
        Compute dλ/dw for the linear wavelength transform.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        dldw : float or ndarray
            Derivative of wavelength with respect to the transformed coordinate.
        """
        return np.ones_like(lam)
