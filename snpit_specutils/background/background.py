from pathlib import Path
import yaml

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import roman_datamodels as rdm
from skimage import measure, morphology

from ..utils import get_package_file
from .result import ModelResult

class Background:
    """
    Background modeling and subtraction for astronomical images.

    This class estimates a scaling factor for a provided (or unity) sky model
    and iteratively refines a sky mask by rejecting outlier pixels (e.g., objects).
    The final scaled model is subtracted from the input science image.

    Parameters
    ----------
    cfgfile : str, optional
        Path to a YAML configuration file containing default parameters.
    **kwargs
        Additional parameters that override values in the configuration file.

    Notes
    -----
    The algorithm:
    1. Initializes a sky mask (all pixels valid unless specified otherwise).
    2. Fits a scale factor (`alpha`) to the background model.
    3. Iteratively updates the sky mask by rejecting high-sigma residuals.
    4. Stops when convergence or stopping conditions are met.
    """

    def __init__(self, cfgfile='background.yaml', **kwargs):
        """
        Initialize the Background object and load configuration parameters.

        Parameters
        ----------
        cfgfile : str, optional
            Path to a YAML configuration file.
        **kwargs
            Parameters that override values from the configuration file.
        """
        self.load_defaults(cfgfile, **kwargs)

    def load_defaults(self, cfgfile, **kwargs):
        """
        Load default parameters from a YAML configuration file.

        Parameters
        ----------
        cfgfile : str
            Path to the YAML configuration file.
        **kwargs
            Key-value pairs that override values from the configuration file.

        Notes
        -----
        The loaded parameters are stored in `self.pars`.
        """

        self.cfgfile = get_package_file(cfgfile)
        print(self.cfgfile)
        with open(self.cfgfile, 'r') as fp:
            self.pars = yaml.safe_load(fp)

        self.pars.update(kwargs)

    def compute_model(self, dat, unc, skymsk):
        """
        Compute the best-fit scaling of the background model.

        Parameters
        ----------
        dat : ndarray
            Science image data.
        unc : ndarray
            Per-pixel uncertainties (same shape as `dat`).
        skymsk : ndarray of bool
            Boolean mask indicating which pixels are considered sky.

        Returns
        -------
        ModelResult
            Named tuple containing fit results:
            (alpha, chi2, npix, chi2nu, resid).

        Notes
        -----
        The fit is performed using inverse-variance weighting.
        Only pixels where both `skymsk` and `self.gpxmsk` are True are used.
        """
        # get all the good pixels
        msk = np.logical_and(skymsk, self.gpxmsk)

        # extract the uncertainty, and weight by inverse variance
        U = unc[msk]
        S = dat[msk] / U
        M = self.mod[msk] / U

        # compute some auxillary values
        Foo = np.sum(S * S)
        Fot = np.sum(S * M)
        Ftt = np.sum(M * M)

        # compute the optimal scaling
        alpha = Fot / Ftt
        chi2 = (Foo - alpha * Fot)
        npix = np.count_nonzero(msk)
        res = (dat - alpha * self.mod) / unc

        return ModelResult(alpha, chi2, npix, chi2 / (npix - 1), res)

    def update_skymask(self, res, footprint, pars):
        """
        Update the sky mask by rejecting object-like pixels.

        Parameters
        ----------
        res : ndarray
            Residual image normalized by uncertainty.
        footprint : ndarray of bool
            Structuring element used for morphological dilation.
        pars : dict
            Parameter dictionary containing thresholds such as `nsigma`
            and `minsize`.

        Returns
        -------
        skymsk : ndarray of bool
            Updated sky mask where True indicates sky pixels.

        Notes
        -----
        Steps:
        1. Identify object pixels using a sigma threshold.
        2. Label connected components.
        3. Remove small objects.
        4. Dilate remaining objects.
        5. Invert to obtain the sky mask.
        """
        objmsk = (pars['nsigma'][1] < res)

        # label the mask
        objlabel = measure.label(objmsk)

        # remove small regions
        objlabel = morphology.remove_small_objects(objlabel, pars['minsize'])

        # grow objmsk
        objmsk = (objlabel != 0)
        objmsk = morphology.binary_dilation(objmsk, footprint=footprint)

        # make back to a skymsk
        skymsk = np.logical_not(objmsk)

        return skymsk


    def makePDF(self, outfile, dat, model, skymsk, alpha, nsigma=1.):
        """
        Make a PDF output of the background-subtracted image

        """

        # make output file
        p = Path(outfile)
        pdffile = p.parent/(p.stem+'.pdf')        
        print(f'Writing PDF: {pdffile}')

        # compute the mask and model
        msk = np.logical_and(skymsk, self.gpxmsk)
        diff = dat-model

        
        # compute stats
        a1, m1, s1 = sigma_clipped_stats(dat[msk])
        vmin1, vmax1 = a1-nsigma*s1, a1+nsigma*s1
        
        a2, m2, s2 = sigma_clipped_stats(diff[msk])
        vmin2, vmax2 = a2-nsigma*s2, a2+nsigma*s2

        # compute ranges        
        norm1 = PowerNorm(gamma=0.5, vmin=vmin1, vmax=vmax1)
        cmap = 'magma'

        # make an output frame
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.08, 1])

        # Colorbar axis
        cax12 = fig.add_subplot(gs[0, :2])
        cax3 = fig.add_subplot(gs[0, 2])

        # Shared axes
        ax0 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[1, 1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[1, 2], sharex=ax0, sharey=ax0)

        # First two: same colormap
        im0 = ax0.imshow(dat, cmap=cmap, norm=norm1)
        im1 = ax1.imshow(model, cmap=cmap, norm=norm1)
        im2 = ax2.imshow(diff, cmap='bwr_r', vmin=vmin2, vmax=vmax2)

        # add some text
        txt = ax1.text(0.05, 0.95, f'$\\alpha$={alpha:.3} DN/s',
            verticalalignment='top', horizontalalignment='left',
            transform=ax1.transAxes)
        txt.set_bbox({'facecolor': 'white', 'alpha': 0.2, 'edgecolor': 'black'})
        
        txt = ax2.text(0.05, 0.95, f'$\\mu$={a2:.2g}$\\pm${s2:.2} DN/s',
            verticalalignment='top', horizontalalignment='left',
            transform=ax2.transAxes)
        txt.set_bbox({'facecolor': 'white', 'alpha': 0.2, 'edgecolor': 'black'})
        
        
    
        # Colorbar for first two
        cbar12 = fig.colorbar(im0, cax=cax12, orientation='horizontal')
        cbar12.set_label("Flux [DN/s]", labelpad=5)
        cbar12.ax.xaxis.set_label_position('top')
        cbar12.ax.xaxis.set_ticks_position('top')

        # Colorbar for third
        cbar3 = fig.colorbar(im2, cax=cax3, orientation='horizontal')
        cbar3.set_label("Residuals [DN/s]", labelpad=5)
        cbar3.ax.xaxis.set_label_position('top')
        cbar3.ax.xaxis.set_ticks_position('top')

        # clean up and write out
        plt.tight_layout()
        plt.savefig(pdffile)

    
    def __call__(self, scifile, skyfile=None, objfile=None,
                 outfile=None, **kwargs):
        """
        Run the full background modeling and subtraction pipeline.

        Parameters
        ----------
        scifile : str
            Path to the input science file (Roman data model).
        skyfile : str, optional
            Path to a FITS file containing a background model.
            If not found or invalid, a unity model is used.
        objfile : str, optional
            Path to an object mask file (currently not implemented).
        outfile : str, optional
            Path to the output file. If None, overwrites `scifile`.
        **kwargs
            Parameters that override configuration values.

        Returns
        -------
        outfile : str
            The name of the output file as written to disk.

        Side Effects
        ------------
        - Subtracts the fitted background model from the input data.
        - Writes the result to `outfile`.

        Notes
        -----
        The method:
        1. Loads data, uncertainty, and data quality arrays.
        2. Loads or constructs a background model.
        3. Iteratively refines a sky mask and fits the model scaling.
        4. Stops when convergence or stopping criteria are met.
        5. Subtracts the scaled model and saves the result.

        Stopping Conditions
        -------------------
        - Convergence of `alpha`.
        - Maximum number of iterations reached.
        - Negative scaling factor.
        - Too few sky pixels remaining.
        """
        pars = self.pars | kwargs
        with rdm.open(scifile) as dm:
            dat = dm.data.astype(float)
            unc = dm.err.astype(float)
            dqa = dm.dq.astype(int)

        # small hack for Ilia sims
        dqa[:, :] = 0
        
        # read the model image
        if isinstance(skyfile, str):
            tmp = Path(skyfile)
            if tmp.exists():
                if tmp.suffix == '.fits':
                    self.mod = fits.getdata(skyfile)
                else:
                    raise NotImplementedError(f"Skyfile type unsupported: {tmp.suffix}")

                # ensure the model is normalized
                self.mod /= self.mod[dat.shape[0] // 2, dat.shape[1] // 2]

                # check the shape
                if self.mod.shape != dat.shape:
                    print("SKY IMAGE IS INVALID SHAPE")
                    skyfile = None
            else:
                print("SKYFILE NOT FOUND.")
                skyfile = None
        else:
            print("NO SKYFILE PRESENT.")
            skyfile = None

        # set to unity if needed
        if skyfile is None:
            self.mod = np.ones_like(dat, dtype=float)

        # read the object mask
        if isinstance(objfile, str):
            raise NotImplementedError('Unknown how to read objmask')
        else:
            skymsk = np.ones_like(dat, dtype=bool)

        # initialize the other masks
        self.gpxmsk = (dqa == 0)

        # create a footprint for dilation
        footprint = np.ones(pars['dilation'], dtype=bool)

        # initialize the algorithm
        result = self.compute_model(dat, unc, skymsk)
        flags = 0
        itr = 0
        message = []

        # start the iterations
        print(f'{itr:4} {result.alpha:.8g} {result.chi2nu:.5g} {flags:2}')
        while not flags:
            skymsk = self.update_skymask(result.resid, footprint, pars)
            newresult = self.compute_model(dat, unc, skymsk)

            # convergence checks
            dalpha = np.abs(newresult.alpha - result.alpha)
            alphathresh = np.abs(newresult.alpha) * pars['epsilon']
            if dalpha < alphathresh:
                message.append("ALPHA CONVERGED.")
                flags += 0b0001

            if itr >= pars['maxiter']:
                message.append("MAXITERS REACHED.")
                flags += 0b0010

            if newresult.alpha <= 0:
                message.append("ALPHA IS NEGATIVE.")
                flags += 0b0100

            if newresult.npix <= pars['minskypix']:
                message.append("TOO FEW SKY PIXELS.")
                flags += 0b1000

            itr += 1
            result = newresult
            print(f'{itr:4} {result.alpha:.8g} {result.chi2nu:.5g} {flags:2}')

        for m in message:
            print(m)

        # additional outputs
        meta = {
            'alpha': result.alpha,
            'chi2': result.chi2,
            'npix': result.npix,
            'chi2nu': result.chi2nu,
            'objfile': objfile,
            'skyfile': skyfile,
            'cfgfile': self.cfgfile,
            'pars': pars
        }

        if outfile is None:
            outfile = scifile

        # make the model
        model = result.alpha * self.mod

        # some outputting functions
        self.makePDF(outfile, dat, model, skymsk, result.alpha)
            
        print(f"SAVING FILE: {outfile}")
        with rdm.open(scifile) as dm:
            # subtract background
            dm.data -= model.astype(dm.data.dtype)

            # add meta data
            dm.snpit = {'background': meta}
            
            # write new asdf
            dm.save(outfile)

        return outfile
