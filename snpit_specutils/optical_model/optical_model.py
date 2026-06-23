import yaml

import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_package_file
from .numba_utils import (disperse_pairwise, disperse, deriv_pairwise, deriv)
from .transformers import (LinearTransformer, LogTransformer)

def validate_inputs(func):
    """
    Custom decorator to validate all inputs in `OpticalModel`
    written by R. Ryan
    Jun 2026
    """
    
    def validate(self, x, y, l, sca, order='1', pairwise=False):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        l = np.atleast_1d(l)
        
        if order not in self.optical['orders']:
            raise KeyError("Invalid spectral order")

        if sca not in self.detector['xy_centers']:
            raise KeyError("Invalid SCA number")

        if pairwise:
            if (x.shape != y.shape) or (x.shape != l.shape):
                raise RuntimeError("Invalid x, y, lam shape for pairwise")
        else:
            if (x.shape != y.shape) or (l.ndim != 1):
                raise RuntimeError("Invalid x, y, lam shape")
        
        return func(self, x, y, l, sca, order=order, pairwise=pairwise)
    
    return validate


class OpticalModel:
    """
    Optical model for the Roman/WFI, optimized to meet the needs
    of the SN PIT.

    This class loads a configuration file describing detector geometry
    and optical polynomial models, and provides methods to compute:

    - Dispersed positions on the Focal Plane Assembly (FPA),
    - Derivatives with respect to wavelength,
    - Local trace normals,
    - Local dispersion scales.

    All intermediate optical coordinates are expressed in Focal Plane
    Assembly (FPA) physical units and mapped to and from Sensor Chip
    Assembly (SCA) pixel coordinates.

    written by R. Ryan, based on code from V. Mehta
    Jun 2026    
    """

    def __init__(self, conffile):
        """
        Initialize the disperser from a configuration file.

        Parameters
        ----------
        conffile : str
            Path to the YAML configuration file.
        """
        
        self.load_conffile(conffile)

        # pre-compile the numba functions
        _, _ = self.disperse(1., 1., 1., 1, pairwise=False)
        _, _ = self.disperse(1., 1., 1., 1, pairwise=True)
        
        _, _ = self.deriv(1., 1., 1., 1, pairwise=False)
        _, _ = self.deriv(1., 1., 1., 1, pairwise=True)
        
        
    def load_conffile(self, conffile):
        """
        Load and parse the disperser configuration file.

        This method reads detector geometry, optical polynomial coefficients,
        and wavelength transformation settings, and precomputes derivative
        coefficients for efficient runtime evaluation.

        Parameters
        ----------
        conffile : str
            Path to the YAML configuration file.
        """

        # obtain the file
        self.conffile = get_package_file(conffile)
        if self.conffile is None:
            raise RuntimeError("No valid config data.")

        # open the file and read the data
        with open(self.conffile, 'r') as fp:
            cfg = yaml.safe_load(fp)

        # fetch the meta data
        self.meta = cfg['meta']

        # set the detector model
        self.detector = cfg['detector_model']

        # change some units
        self.detector['pixel_scale'] /= 3600.
        
        # set the optical model
        self.optical = {k: v for k, v in cfg['optical_model'].items()}

        # process each spectral order
        self.optical['orders'] = {}
        for order, data in cfg['optical_model']['orders'].items():
            
            # extract the parameters
            Xij = np.asarray(data['xmap_ij_coeff'])
            Yij = np.asarray(data['ymap_ij_coeff'])
            Cijk = np.asarray(data['ids_ijk_coeff'])
            Dijk = np.asarray(data['crv_ijk_coeff'])

            # check that the data are valid
            if Xij.all() and Yij.all() and Cijk.all() and Dijk.all():
                
                # compute derivatives
                dCijk = self.deriv_coeffs(Cijk)
                dDijk = self.deriv_coeffs(Dijk)

                # save the results
                self.optical['orders'][order] = {'Xij': Xij,
                                                'Yij': Yij,
                                                'Cijk': Cijk,
                                                'dCijk': dCijk,
                                                'Dijk': Dijk,
                                                'dDijk': dDijk}

        # transform the wavelengths            
        transform = self.optical['wl_transform'].lower()
        if transform == 'log':
            self.lam_transformer = LogTransformer(self.optical['wl_reference'])
        elif transform == 'linear':
            self.lam_transformer = LinearTransformer(self.optical['wl_reference'])
        else:
            raise NotImplementedError(f'Invalid transform {transform}')
            

    @staticmethod
    def deriv_coeffs(M):
        """
        Compute derivative polynomial coefficients with respect to the first
        dimension using Horner-compatible form.

        Parameters
        ----------
        M : ndarray, shape (n, ...)
            Polynomial coefficient tensor where the first axis corresponds
            to the variable of differentiation.

        Returns
        -------
        dM : ndarray, shape (n-1, ...)
            Derivative coefficient tensor.
        """
        n = M.shape[0]
        ii = np.arange(1, n, dtype=float)
        shape = (n-1,) + (1,) * (M.ndim - 1)
        dM = M[1:] * ii.reshape(shape)

        return dM


    def sca_to_fpa(self, xsca, ysca, sca):
        """
        Convert Sensor Chip Assembly (SCA) pixel coordinates to
        Focal Plane Assembly (FPA) physical coordinates.

        Parameters
        ----------
        xsca, ysca : array_like
            Pixel coordinates on the SCA detector.
        sca : int or key
            Identifier for the detector segment.

        Returns
        -------
        xfpa, yfpa : ndarray
            Physical coordinates in the Focal Plane Assembly (FPA) frame.
        """
        dx = self.detector['crpix1']-xsca # note the negative
        dy = ysca-self.detector['crpix2']

        xcen, ycen = self.detector['xy_centers'][sca]
         
        xfpa = (xcen*self.detector['plate_scale'] + dx)*self.detector['pixel_scale']
        yfpa = (ycen*self.detector['plate_scale'] + dy)*self.detector['pixel_scale']

        return xfpa, yfpa

    
    def mpa_to_sca(self, xmpa, ympa, sca):
        """
        Convert Mosaic Plate Assembly (MPA) physical coordinates back to
        Sensor Chip Assembly (SCA) pixel coordinates.

        Parameters
        ----------
        xmpa, ympa : array_like
            Physical Mosaic Plate Assembly coordinates.
        sca : int or key
            Identifier for the detector segment.

        Returns
        -------
        xsca, ysca : ndarray
            Detector pixel coordinates.
        """
        xcen, ycen = self.detector['xy_centers'][sca]
        
        xoff = (xmpa - xcen)*self.detector['plate_scale']
        yoff = (ympa - ycen)*self.detector['plate_scale']

        xsca = self.detector['crpix1'] - xoff # note the negative
        ysca = self.detector['crpix2'] + yoff

        return xsca, ysca

    @validate_inputs
    def disperse(self, x0, y0, lam, sca, order='1', pairwise=False):
        """
        Compute dispersed detector coordinates for given source positions
        and wavelengths.

        Parameters
        ----------
        x0, y0 : array_like
            Undispersed Sensor Chip Assembly (SCA) pixel coordinates.
        lam : array_like
            Physical wavelengths.
        sca : int or key
            Detector segment identifier.
        order : str, optional
            Spectral order identifier (default is '1').
        pairwise : bool, optional
            If True, treat x0, y0, and lam as pairwise-aligned arrays.
            If False, compute a full grid over x0/y0 and lam.

        Returns
        -------
        xp, yp : ndarray
            Dispersed SCA pixel coordinates.
        """
        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)
                
        # transform wavelengths
        wtran = self.lam_transformer.evaluate(lam)
        
        if pairwise:
            xt, yt = disperse_pairwise(
                self.optical['orders'][order]['Xij'],
                self.optical['orders'][order]['Yij'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['Dijk'],
                xfpa, yfpa, wtran)
        else:
            xt, yt = disperse(
                self.optical['orders'][order]['Xij'],
                self.optical['orders'][order]['Yij'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['Dijk'],
                xfpa, yfpa, wtran)
            
            
        # convert back to SCA coordinates (this method should deal with
        # the traces that span multiple SCAs)
        xp, yp = self.mpa_to_sca(xt, yt, sca)
       
        return xp, yp

    @validate_inputs
    def deriv(self, x0, y0, lam, sca, order='1', pairwise=False):
        """
        Compute derivatives of dispersed detector coordinates with respect
        to wavelength.

        Parameters
        ----------
        x0, y0 : array_like
            Undispersed Sensor Chip Assembly (SCA) pixel coordinates.
        lam : array_like
            Physical wavelengths.
        sca : int or key
            Detector segment identifier.
        order : str, optional
            Spectral order identifier (default is '1').
        pairwise : bool, optional
            If True, treat x0, y0, and lam as pairwise-aligned arrays.
            If False, compute a full grid over x0/y0 and lam.

        Returns
        -------
        dxdl, dydl : ndarray
            Derivatives of dispersed SCA pixel coordinates with respect to
            wavelength.
        """
        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)
                
        # transform wavelengths
        wtran = self.lam_transformer.evaluate(lam)
        dldw = self.lam_transformer.deriv(lam)
        
        if pairwise:
            dxdw, dydw = deriv_pairwise(
                self.optical['orders'][order]['dCijk'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['dDijk'],
                xfpa, yfpa, wtran)

            dxdl = dxdw/dldw
            dydl = dydw/dldw
            
        else:
            dxdw, dydw = deriv(
                self.optical['orders'][order]['dCijk'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['dDijk'],
                xfpa, yfpa, wtran)

            dxdl = dxdw/dldw[np.newaxis, :]
            dydl = dydw/dldw[np.newaxis, :]

        dxdl *= self.detector['plate_scale']
        dydl *= self.detector['plate_scale']
        
        return dxdl, dydl

    def normal(self, *args, **kwargs):
        """
        Compute unit-normal vectors to the spectral trace.

        This returns vectors perpendicular to the dispersion direction
        at each sampled point.

        Returns
        -------
        nx, ny : ndarray
            Components of the normal vectors.
        """
        dxdl, dydl = self.deriv(*args, **kwargs)
        return dydl, -dxdl
            
    def dispersion(self, *args, **kwargs):
        """
        Compute the local dispersion scale (dλ/dr) along the spectral trace.

        Returns
        -------
        dldr : ndarray
            Local dispersion (wavelength per pixel).
        """
        dxdl, dydl = self.deriv(*args, **kwargs)
        dldr = 1./np.hypot(dxdl, dydl)
        return dldr

