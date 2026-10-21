from functools import reduce
import glob
import inspect
import itertools
from operator import getitem
import os
import sys

import numpy as np
from regions import Regions
try:
    import progressbar
except: # pylint: disable=bare-except
    print('Please install the progressbar2 module (not progressbar)')
    sys.exit()
import astropy.units as u
from matplotlib import pyplot

from pybkgmodel.data import RunSummary
from pybkgmodel.message import message
from pybkgmodel.model import (WobbleMap,
                              ExclusionMap
                            )
from pybkgmodel.camera import RectangularCameraImage, CAMERA_RESPONSE_PARAM_NAMES, super_gaussian

# Order of the camera_response() fit parameters, as expected by
# CameraImage.fit_camera_response() / fitted_differential_rate().
# sigma_core_lo, sigma_core_hi, e_core_lo, e_core_hi, x0_core, y0_core:
#   Gaussian core (camera acceptance shape's central component): its
#   width at the two ends of the fitted energy range, and its
#   (energy-independent) center.
# sigma_tail, p_tail, ecc_tail, phi_tail: elliptical super-Gaussian
#   tail (extended halo component), fixed at the camera center.
# w0, delta_w: fraction of the spatial distribution's weight in the
#   core, and its energy dependence.
# (the linear function of y has both its slope and intercept fixed,
#   i.e. no y-dependence, for now; see camera_response())
# norm, e0, alpha, beta: log-parabola function of energy.
def default_camera_response_guess(bkg_map):
    """
    Generic initial guess and bounds for CameraImage.fit_camera_response(),
    derived only from the binning and the overall count/exposure scale of
    the given background map (no knowledge of the actual acceptance shape
    is assumed).

    Parameters
    ----------
    bkg_map: pybkgmodel.camera.CameraImage
        Background map to be fit.

    Returns
    -------
    p0: list
        Initial guess for the fit parameters.
    bounds: list of (min, max)
        Bounds for the fit parameters.
    """
    x_min, x_max = bkg_map.xedges.min().to_value(u.deg), bkg_map.xedges.max().to_value(u.deg)
    y_min, y_max = bkg_map.yedges.min().to_value(u.deg), bkg_map.yedges.max().to_value(u.deg)
    x_range = x_max - x_min
    e0_guess = 1.0

    # The core and tail widths are kept in separate ranges, split at a
    # quarter of the FOV extent: the core is the narrower component,
    # the tail the wider one. Without this, the fit can swap the two
    # roles (a Gaussian wider than the FOV acting as a flat pedestal,
    # and a narrow tail forming the peak), which leaves the
    # core center (x0_core, y0_core) unconstrained by the data.
    sigma_split = x_range / 4

    # The core width is given at the two ends of the energy range
    # constrained by the data (fixed, see fit_energy_range()), with a
    # power law in between (see core_width()). Bounding both widths
    # below sigma_split keeps the core narrower than the tail over that
    # whole range, while letting its width change with energy.
    _, e_core_lo0, e_core_hi0 = bkg_map.fit_energy_range()
    sigma_core_lo0 = sigma_core_hi0 = sigma_split / 2
    x0_core0, y0_core0 = 0.0, 0.0

    # The tail exponent p_tail >= 1 ranges from a Gaussian (1) to a
    # flat-topped plateau with a steep edge, as the camera acceptance
    # often shows at high energies.
    sigma_tail0 = x_range / 2
    p_tail0 = 1.5
    ecc_tail0 = 0.1
    phi_tail0 = 0.0

    # w0 (the core fraction at e = e0) is fixed to 0.5, which makes e0
    # the energy at which the core and tail contribute equally (see
    # core_fraction()). This removes the degeneracy of the reference
    # energy e0, which could otherwise be traded against w0, norm,
    # alpha and beta without changing the model.
    w0 = 0.5
    delta_w0 = -1.0

    alpha0 = 1.7
    beta0 = 0.1

    # Rough normalization guess, so that the model peak count matches
    # the observed peak count for the initial guess of the shape
    # parameters above (both components centered at the origin). The
    # linear(y) function's intercept is fixed to 1 inside
    # camera_response() (see linear()), so it does not enter here.
    core_peak = super_gaussian(0.0, 0.0, sigma_core_lo0, p=1.0)
    tail_peak = super_gaussian(0.0, 0.0, sigma_tail0, p=p_tail0)
    spatial_peak = w0*core_peak + (1 - w0)*tail_peak
    exposure0 = bkg_map.raw_exposure[bkg_map.mask].mean().to_value(u.s)
    counts_peak = bkg_map.raw_counts.max()
    norm0 = max(counts_peak / max(spatial_peak * exposure0, 1e-30), 1e-10)

    p0 = [
        sigma_core_lo0, sigma_core_hi0, e_core_lo0, e_core_hi0, x0_core0, y0_core0,
        sigma_tail0, p_tail0, ecc_tail0, phi_tail0,
        w0, delta_w0,
        norm0, e0_guess, alpha0, beta0,
    ]

    bounds = [
        (x_range / 100, sigma_split),      # sigma_core_lo (narrower than the tail)
        (x_range / 100, sigma_split),      # sigma_core_hi (narrower than the tail)
        (e_core_lo0, e_core_lo0),          # e_core_lo (fixed, see above)
        (e_core_hi0, e_core_hi0),          # e_core_hi (fixed, see above)
        (x_min, x_max),                    # x0_core
        (y_min, y_max),                    # y0_core
        (sigma_split, x_range),            # sigma_tail (wider than the core, capped
                                            # at the FOV extent: wider halos aren't
                                            # constrained by the data)
        (1.0, 5.0),                        # p_tail
        (0.0, 0.9),                        # ecc_tail
        (-np.pi / 2, np.pi / 2),           # phi_tail
        (w0, w0),                          # w0 (fixed, see above)
        (-10.0, -0.001),                       # delta_w
        (norm0 / 1e3, norm0 * 1e3),        # norm
        (0.9, 10.0),                       # e0 (core/tail crossover energy, TeV)
        (0.0, 3.0),                      # alpha
        (-1.0, 1.0),                       # beta
    ]

    return p0, bounds

# Starting values of the core/tail crossover energy e0 (TeV) and of
# the tail exponent p_tail tried by fit_camera_response_multistart():
# the fit has several local minima (e.g. with the tail taking over the
# core's role in part of the energy range), and which one a single
# L-BFGS-B run ends in depends on its starting point.
MULTISTART_E0 = (1.0, 3.0, 8.0)
MULTISTART_P_TAIL = (1.5, 3.0)


def fit_camera_response_multistart(bkg_map, p0, bounds):
    """
    Fit camera_response() to the background map with L-BFGS-B from
    several starting points, varying e0 and p_tail over MULTISTART_E0
    and MULTISTART_P_TAIL (clipped to their bounds), the other
    parameters starting at p0. Returns the converged fit with the
    lowest fit statistic.

    Parameters
    ----------
    bkg_map: pybkgmodel.camera.CameraImage
        Background map to be fit.
    p0: list
        Initial guess for the other fit parameters, see
        default_camera_response_guess().
    bounds: list of (min, max)
        Bounds for the fit parameters.

    Returns
    -------
    result: scipy.optimize.OptimizeResult or None
        Best converged fit (see CameraImage.fit_camera_response()), or
        None if no start converged to a finite fit statistic.
    """
    i_e0 = CAMERA_RESPONSE_PARAM_NAMES.index('e0')
    i_p = CAMERA_RESPONSE_PARAM_NAMES.index('p_tail')

    best = None
    for e0, p_tail in itertools.product(MULTISTART_E0, MULTISTART_P_TAIL):
        start = list(p0)
        start[i_e0] = np.clip(e0, *bounds[i_e0])
        start[i_p] = np.clip(p_tail, *bounds[i_p])

        try:
            # The default L-BFGS-B evaluation limit (15000) can be hit
            # just before convergence, with the parameters already
            # settled, so it is raised here.
            result = bkg_map.fit_camera_response(
                start, bounds=bounds, method='L-BFGS-B',
                options=dict(maxfun=50000, maxiter=50000)
            )
        except Exception:  # pylint: disable=broad-except
            continue

        if result.success and np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    return best


# Prior weight used to combine the camera_response() fit (as the
# prior) with the observed counts (as the data) into the Bayesian
# posterior background model written out by write_maps(); see
# CameraImage.posterior_counts(). It is given per unit solid angle:
# each pixel's prior weight k (in equivalent prior "pseudo-counts") is
# this density times the pixel's solid angle (see
# posterior_prior_strength()). A per-pixel constant k would make the
# data weigh less the finer the binning, as the counts per pixel scale
# with the pixel size while k would not. The reference is k = 25 for
# a 0.3 deg x 0.3 deg pixel (k ~ 2.8 for 0.1 deg pixels).
POSTERIOR_PRIOR_DENSITY = 25.0 / (0.3 * u.deg)**2


def posterior_prior_strength(bkg_map):
    """
    Per-pixel prior weight k of the Bayesian posterior (see
    CameraImage.posterior_counts()) for the given background map:
    POSTERIOR_PRIOR_DENSITY times each pixel's solid angle.

    Parameters
    ----------
    bkg_map: pybkgmodel.camera.CameraImage
        Background map.

    Returns
    -------
    k: numpy.ndarray
        Prior weight in equivalent prior "pseudo-counts", of the shape
        of the camera pixels (nx, ny).
    """
    return (POSTERIOR_PRIOR_DENSITY * bkg_map.pixel_area).decompose().value

# Neighbourhood of the moving average applied to the posterior
# background model written out by write_maps(): pixels whose centers,
# seen from the fitted Gaussian core peak, are within
# SMOOTHING_MAX_DPHI in azimuth and SMOOTHING_MAX_DR in radial
# distance (counted as negative across the peak, see
# pybkgmodel.camera.polar_neighbourhood()), or within
# SMOOTHING_MAX_DIST of each other (see below), are
# averaged together, then corrected with the fitted model
# shape and renormalized to the posterior total per energy bin (see
# CameraImage.smoothed_posterior_rate()).
SMOOTHING_MAX_DPHI = 10 * u.deg
SMOOTHING_MAX_DR = 0.25 * u.deg
# Pixels whose centers are within this distance of each other are
# always averaged together, whatever their azimuth and radial distance.
# This keeps the neighbourhood from shrinking to a thin sliver near the
# core peak, which otherwise leaves a radial pattern there.
SMOOTHING_MAX_DIST = 0.25 * u.deg

# list of class attributes, which have a unit assigned
quantity_list = [
                'time_delta',
                'pointing_delta',
                'x_min',
                'x_max',
                'y_min',
                'y_max',
                'e_min',
                'e_max'
                ]

# dictionary to map names in the config file to the class attribute names
config_class_map = {
    'files' : ['data', 'mask'],
    'cuts' : ['data', 'cuts'],
    'out_dir' : ['output', 'directory'],
    'out_prefix' : ['output', 'prefix'],
    'overwrite' : ['output', 'overwrite'],
    'time_delta' : ['run_matching', 'time_delta'],
    'pointing_delta' : ['run_matching', 'pointing_delta'],
    'x_min' : ['binning', 'x', 'min'],
    'x_max' : ['binning', 'x', 'max'],
    'y_min' : ['binning', 'y', 'min'],
    'y_max' : ['binning', 'y', 'max'],
    'x_nbins' : ['binning', 'x', 'nbins'],
    'y_nbins' : ['binning', 'y', 'nbins'],
    'e_min' : ['binning', 'energy', 'min'],
    'e_max' : ['binning', 'energy', 'max'],
    'e_nbins' : ['binning', 'energy', 'nbins'],
    'excl_region' : ['exclusion_regions']
}

class BkgMakerBase:
    """
    Base class for all processing classes, which store the settings from the
    configuation file and facilitate the generation of background maps.
    Not intended for direct usage.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(
                self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ) -> None:

        """
        Function initializing a processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis

         Returns
        -------
        out
            processing object
        """

        self.files          = glob.glob(files)
        self.runs           = tuple(
                                filter(
                                    lambda r: r.obs_id is not None,
                                    [RunSummary(fname) for fname in
                                    self.files]
                                    )
                                )
        self.cuts           = cuts

        self.out_dir        = out_dir
        self.out_prefix     = out_prefix
        self.overwrite      = overwrite

        self.x_edges        = np.linspace(
                                x_min,
                                x_max,
                                x_nbins+1
                                )

        self.y_edges        = np.linspace(
                                y_min,
                                y_max,
                                y_nbins+1
                                )

        self.e_edges        = np.geomspace(
                                e_min,
                                e_max,
                                e_nbins+1
                                )

        self._bkg_maps   = {}

        self._bkg_map_maker = None


    @property
    def bkg_map_maker(self):
        """Getter for bkg_map_maker."""
        print("This class uses the background method:",
              self.__bkg_map_maker.__class__.__name__)
        return self._bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, value):
        """Setter for bkg_map_maker."""
        self._bkg_map_maker = value

    @property
    def bkg_maps(self):
        """Getter for bkg_maps."""
        return self._bkg_maps

    @classmethod
    def from_config_file(cls, config):
        """
        Function initializing a prozessing object from an input dictionary.

        Parameters
        ----------
        config : dict
            dictionary containing the settings read from the yaml configuration
            file.

        Raises
        ------
        ValueError
            Error is raised if no input dictionary is provided.
        """

        if config is None:
            raise ValueError(
                "No configuration file provided."
            )

        # obtain the parameters of the class on runtime; not know apriori
        class_params = inspect.signature(cls).parameters

        params_for_init = {}

        # Fill dictionary of parameters required by the corresponding class
        # with the values from the config file dictionary
        for current_par in class_params:

            # read required class parameters from the config file dictionary
            try:
                current_par_val = reduce(
                    getitem,
                    config_class_map[f"{current_par}"],
                    config
                    )

                if current_par in quantity_list:
                    current_par_val = u.Quantity(current_par_val)
                else:
                    pass

            except KeyError:
                print(
                    f"Parameter {config_class_map[f'{current_par}']} missing in config file."
                    )

            # assign the extracted parameter to the dictionary from which the
            # class object will be created
            params_for_init[f"{current_par}"] = current_par_val

        return cls(**params_for_init)

    def generate_runwise_maps(self) -> dict:
        """
        Returns a dictionary containing the runwise bkg maps and output file names
        for each input run.

        Returns
        -------
        dict
            {'maps', 'outnames'}
        """
        maps = {}

        with progressbar.ProgressBar(max_value=len(self.runs)) as progress:
            for run_idx, run in enumerate(self.runs):

                # Here the corrsponding bkg reconstruction algorith is applied
                # to obtain the runwise bkg map
                bkg_map = self._bkg_map_maker.get_runwise_bkg(target_run = run)

                # get corresponding names for the bkg maps under which they can
                # be safed
                base_name = os.path.basename(run.file_name)
                base_name, _ = os.path.splitext(base_name)

                output_name = os.path.join(
                                        self.out_dir,
                                        f"{self.out_prefix}{base_name}.fits"
                                        )

                maps[f'{output_name}'] = bkg_map

                progress.update(run_idx)

        self._bkg_maps = maps

        return maps

    @staticmethod
    def save_diagnostic_plots(bkg_map, base_path, result=None, prior_strength=None):
        """
        Save diagnostic plot images for a single background map: the
        observed rate map (CameraImage.plot()) for every energy bin,
        and, if a camera_response() fit result is available, the
        spatial (CameraImage.plot_fit_check()) and spectral
        (CameraImage.plot_fit_spectrum()) checks of the Bayesian
        posterior counts (see CameraImage.posterior_counts()) against
        the data.

        Parameters
        ----------
        bkg_map : pybkgmodel.camera.CameraImage
            Background map to plot.
        base_path : str
            Output path without extension. '_map_bin<i>.png',
            '_fit_check_bin<i>.png', '_counts_smoothing_bin<i>.png',
            '_fit_spectrum.png', '_core_fraction.png' and
            '_smoothing_neighbourhood.png' are appended to it.
        result : scipy.optimize.OptimizeResult, optional
            Result of bkg_map.fit_camera_response(), must have a
            model_counts attribute, used as the posterior_counts()
            prior. If None (default), only the plain rate map is
            saved.
        prior_strength : float or numpy.ndarray, optional
            Prior weight passed to plot_fit_check() /
            plot_fit_spectrum() (see CameraImage.posterior_counts()).
            Defaults to posterior_prior_strength(bkg_map).
        """
        if prior_strength is None:
            prior_strength = posterior_prior_strength(bkg_map)

        n_ebins = len(bkg_map.energy_edges) - 1

        for i in range(n_ebins):
            bkg_map.plot(energy_bin_id=i)
            pyplot.savefig(f"{base_path}_map_bin{i}.png")
            pyplot.close(pyplot.gcf())

            if result is not None:
                fig, _ = bkg_map.plot_smoothing_neighbourhood(
                    result, max_dphi=SMOOTHING_MAX_DPHI, max_dr=SMOOTHING_MAX_DR, max_dist=SMOOTHING_MAX_DIST,
                    energy_bin_id=i
                )
                fig.savefig(f"{base_path}_counts_smoothing_bin{i}.png", bbox_inches='tight')
                pyplot.close(fig)

                fig, _ = bkg_map.plot_fit_check(
                    result, energy_bin_id=i, prior_strength=prior_strength,
                    smoothing=dict(
                        max_dphi=SMOOTHING_MAX_DPHI, max_dr=SMOOTHING_MAX_DR, max_dist=SMOOTHING_MAX_DIST
                    )
                )
                fig.savefig(f"{base_path}_fit_check_bin{i}.png")
                pyplot.close(fig)

        if result is not None:
            fig, _ = bkg_map.plot_fit_spectrum(result, prior_strength=prior_strength)
            fig.savefig(f"{base_path}_fit_spectrum.png")
            pyplot.close(fig)

            fig, _ = bkg_map.plot_core_fraction(result)
            fig.savefig(f"{base_path}_core_fraction.png")
            pyplot.close(fig)

            fig, _ = bkg_map.plot_smoothing_neighbourhood(
                result, max_dphi=SMOOTHING_MAX_DPHI, max_dr=SMOOTHING_MAX_DR, max_dist=SMOOTHING_MAX_DIST
            )
            # The legend sits below the axes; keep it in the image.
            fig.savefig(f"{base_path}_smoothing_neighbourhood.png", bbox_inches='tight')
            pyplot.close(fig)

    @staticmethod
    def write_maps(bkg_maps, overwrite) -> None:
        """
        This method writes the generated bkgmaps to the corresponding output
        path. The output follows the definition in
        https://gamma-astro-data-formats.readthedocs.io/

        Prior to writing, a camera_response() model (see
        pybkgmodel.camera.CameraImage.fit_camera_response()) is fit to
        each map's observed counts (from several starting points, see
        fit_camera_response_multistart()), and used as the prior for a
        bin-by-bin Bayesian update against the observed counts (see
        CameraImage.posterior_counts(), with prior weight
        POSTERIOR_PRIOR_DENSITY per unit solid angle, see
        posterior_prior_strength()). The resulting posterior rate is
        what gets written out as the BKG column. If the fit does not
        converge, the map falls back to the raw counts-based rate
        (CameraImage.differential_rate()) and a warning is printed.

        If the fit succeeded, the posterior is additionally smoothed
        before writing, by a moving average over the pixels within
        SMOOTHING_MAX_DPHI in azimuth and SMOOTHING_MAX_DR in radial
        distance, seen from the fitted Gaussian core peak, or within
        SMOOTHING_MAX_DIST of the pixel itself, corrected
        with the fitted model shape and renormalized to the posterior
        total per energy bin (see CameraImage.smoothed_posterior_rate()).

        Diagnostic plot images (the rate map, and, if the fit
        succeeded, the posterior fit-quality checks) are saved
        alongside each output FITS file, see save_diagnostic_plots().

        Parameters
        ----------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.

        overwrite:  bool
            Whether to overwrite existing output files of same name.
        """

        for key, bkg_map in bkg_maps.items():
            p0, bounds = default_camera_response_guess(bkg_map)

            dnde = None
            fit_result = fit_camera_response_multistart(bkg_map, p0, bounds)

            if fit_result is None:
                # L-BFGS-B can struggle when the fit parameters span very
                # different scales; retry with the more robust (but
                # slower) gradient-free Nelder-Mead before giving up.
                try:
                    fit_result = bkg_map.fit_camera_response(
                        p0, bounds=bounds, method='Nelder-Mead',
                        options=dict(maxiter=50000, maxfev=50000, adaptive=True)
                    )
                    if not fit_result.success:
                        fit_result = None
                except Exception:  # pylint: disable=broad-except
                    fit_result = None

            if fit_result is None:
                message(
                    f"camera_response fit did not converge for '{key}'; "
                    "falling back to the raw counts-based background rate."
                )
            else:
                message(f"Best-fit camera_response parameters for '{key}':")
                for name, val in zip(CAMERA_RESPONSE_PARAM_NAMES, fit_result.x):
                    print(f"    {name:12s} = {val: .6g}")

            base_path, _ = os.path.splitext(key)
            BkgMakerBase.save_diagnostic_plots(bkg_map, base_path, result=fit_result)

            if fit_result is not None:
                dnde = bkg_map.smoothed_posterior_differential_rate(
                    fit_result, prior_strength=posterior_prior_strength(bkg_map),
                    max_dphi=SMOOTHING_MAX_DPHI, max_dr=SMOOTHING_MAX_DR, max_dist=SMOOTHING_MAX_DIST
                )

            bkg_map.to_hdu(bkg_rate=dnde).writeto(key, overwrite=overwrite)

class Runwise(BkgMakerBase):
    """
    Class defining common functions for the runwise processing classes.
    Not intended for direct usage.
    """

    def get_maps(self):
        """ Method for generating and saving runwise background maps to the
        output file.

        Returns
        -------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.
        """

        self.generate_runwise_maps()
        self.write_maps(bkg_maps=self.bkg_maps, overwrite=self.overwrite)
        return self.bkg_maps

class Stacked(BkgMakerBase):
    """
    Class defining common functions for the stacked processing classes.
    Not intended for direct usage.
    """

    def get_maps(self):
        """ Method for generating and saving stacked background maps to the
        output file.

        Returns
        -------
        bkg_maps : dict
            Dictionary containing the stacked bkg map and output name.
        """

        # Each run is read, masked and counted exactly once. Stacking
        # the runwise maps instead would count every run once per map
        # it is a neighbour in, so the stacked counts would not follow
        # Poisson statistics, and runs with more neighbours would weigh
        # more.
        with progressbar.ProgressBar(max_value=len(self.runs)) as progress:
            def runs():
                for run_idx, run in enumerate(self.runs):
                    yield run
                    progress.update(run_idx + 1)

            stacked_map = self._bkg_map_maker.get_stacked_bkg(runs=runs())

        stacked_name = os.path.join(
                self.out_dir,
                f"{self.out_prefix}stacked_bkg_map.fits"
                )
        self._bkg_maps = {stacked_name: stacked_map}
        self.write_maps(bkg_maps=self.bkg_maps, overwrite=self.overwrite)
        return self.bkg_maps

class RunwiseWobbleMap(Runwise):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of runwise background maps using the wobble
    background method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a runswise wobble map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )

        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.bkg_map_maker = WobbleMap(runs=self.runs,
                                        x_edges=self.x_edges,
                                        y_edges=self.y_edges,
                                        e_edges=self.e_edges,
                                        cuts=self.cuts,
                                        time_delta=self.time_delta,
                                        pointing_delta=self.pointing_delta
                                        )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, WobbleMap):
            raise TypeError(f"Maker must be of type {WobbleMap}")
        super(RunwiseWobbleMap, type(self)).bkg_map_maker.__set__(self, maker)

class StackedWobbleMap(Stacked):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of a stacked background map using the wobble
    background method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a stacked wobble map processing object.

       Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.bkg_map_maker  = WobbleMap(runs=self.runs,
                                        x_edges=self.x_edges,
                                        y_edges=self.y_edges,
                                        e_edges=self.e_edges,
                                        cuts=self.cuts,
                                        time_delta=self.time_delta,
                                        pointing_delta=self.pointing_delta
                                        )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, WobbleMap):
            raise TypeError(f"Maker must be of type {WobbleMap}")
        super(StackedWobbleMap, type(self)).bkg_map_maker.__set__(self, maker)

class RunwiseExclusionMap(Runwise):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of runwise background maps using the exclusion
    map method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    excl_region : list
        List of regions to be excluded from the background map in ds9
        format.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                excl_region,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a runswise exclusion map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        excl_region : list
            List of regions to be excluded from the background map in ds9
            format.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in
                               excl_region]
        self.bkg_map_maker  = ExclusionMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           regions=self.excl_region,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, ExclusionMap):
            raise TypeError(f"Maker must be of type {ExclusionMap}")
        super(RunwiseExclusionMap, type(self)).bkg_map_maker.__set__(self, maker)
        
class StackedExclusionMap(Stacked):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of a stacked background map using the exclusion
    map method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    excl_region : list
        List of regions to be excluded from the background map in ds9
        format.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                excl_region,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a stacked exclusion map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        excl_region : list
            List of regions to be excluded from the background map in ds9
            format.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in
                               excl_region]
        self.bkg_map_maker  = ExclusionMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           regions=self.excl_region,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, ExclusionMap):
            raise TypeError(f"Maker must be of type {ExclusionMap}")
        super(StackedExclusionMap, type(self)).bkg_map_maker.__set__(self, maker)
