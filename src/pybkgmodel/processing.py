from functools import reduce
import glob
import inspect
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
from pybkgmodel.camera import RectangularCameraImage

# Order of the camera_response() fit parameters, as expected by
# CameraImage.fit_camera_response() / fitted_differential_rate().
# sigma0, delta_sigma, gamma, ecc, phi: 2D King function (camera
#   acceptance shape); sigma0/delta_sigma set its energy-dependent width.
# a0, delta_a: linear function of y; a0/delta_a set its energy-dependent
#   slope.
# norm, e0, alpha, beta: log-parabola function of energy.
# eth, s: low-energy cutoff function of energy.
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
    x_range = (bkg_map.xedges.max() - bkg_map.xedges.min()).to_value(u.deg)
    e_min = bkg_map.energy_edges.min()
    e_max = bkg_map.energy_edges.max()
    e0_guess = np.sqrt(e_min * e_max).to_value(u.TeV)
    eth_guess = e_min.to_value(u.TeV)

    sigma0 = x_range / 4
    delta_sigma0 = 0.0
    gamma0 = 2.0
    ecc0 = 0.0
    phi0 = 0.0
    a0 = 0.0
    delta_a0 = 0.0
    alpha0 = 2.0
    beta0 = 0.1
    s0 = 2.0

    # Rough normalization guess, so that the model peak count matches
    # the observed peak count for the initial guess of the shape
    # parameters above (sigma(e0) = sigma0, a(e0) = a0). The linear(y)
    # function's intercept is fixed to 1 inside camera_response() (see
    # linear()), so it does not enter here.
    king_peak = (1 - 1/gamma0) / (2 * np.pi * sigma0**2)
    cutoff0 = np.exp(-(eth_guess / e0_guess)**s0)
    exposure0 = bkg_map.raw_exposure[bkg_map.mask].mean().to_value(u.s)
    counts_peak = bkg_map.raw_counts.max()
    norm0 = max(counts_peak / max(king_peak * cutoff0 * exposure0, 1e-30), 1e-10)

    p0 = [
        sigma0, delta_sigma0, gamma0, ecc0, phi0,
        a0, delta_a0,
        norm0, e0_guess, alpha0, beta0,
        eth_guess, s0
    ]

    bounds = [
        (sigma0 / 10, sigma0 * 10),        # sigma0
        (-3.0, 3.0),                       # delta_sigma
        (1.01, 10.0),                      # gamma
        (0.0, 0.9),                        # ecc
        (-np.pi / 2, np.pi / 2),           # phi
        (-1.0, 1.0),                       # a0
        (-3.0, 3.0),                       # delta_a
        (norm0 / 1e3, norm0 * 1e3),        # norm
        (e0_guess * 0.99, e0_guess * 1.01),  # e0 (effectively fixed)
        (-2.0, 5.0),                       # alpha
        (-2.0, 2.0),                       # beta
        (eth_guess * 0.3, eth_guess * 3),  # eth
        (0.3, 10.0),                       # s
    ]

    return p0, bounds

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
    def stack_maps(bkg_maps, x_edges, y_edges, e_edges) -> RectangularCameraImage:
        """
        Returns a stacked bkg map of all the runs.

        Parameters
        ----------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.

        x_edges : np.ndarray
            Array of the bin edges along the x/azimuth axis; linear binning.

        y_edges : np.ndarray
            Array of the bin edges along the y/Zenith axis; linear binning.

        e_edges : np.ndarray
            Array of the bin edges in energy; logarithmic binning.

        Returns
        -------
        RectangularCameraImage
        """
        counts = np.sum([m.counts for m in bkg_maps.values()],
                           axis=0)
        exposure = u.Quantity([m.exposure for m in bkg_maps.values()]
                              ).sum(axis=0)
        stacked_map = RectangularCameraImage(counts,
                                             x_edges,
                                             y_edges,
                                             e_edges,
                                             exposure=exposure)
        return stacked_map

    @staticmethod
    def save_diagnostic_plots(bkg_map, base_path, result=None):
        """
        Save diagnostic plot images for a single background map: the
        observed rate map (CameraImage.plot()) for every energy bin,
        and, if a camera_response() fit result is available, the
        spatial (CameraImage.plot_fit_check()) and spectral
        (CameraImage.plot_fit_spectrum()) fit-quality checks.

        Parameters
        ----------
        bkg_map : pybkgmodel.camera.CameraImage
            Background map to plot.
        base_path : str
            Output path without extension. '_map_bin<i>.png',
            '_fit_check_bin<i>.png' and '_fit_spectrum.png' are
            appended to it.
        result : scipy.optimize.OptimizeResult, optional
            Result of bkg_map.fit_camera_response() /
            fitted_differential_rate(), must have a model_counts
            attribute. If None (default), only the plain rate map is
            saved.
        """
        n_ebins = len(bkg_map.energy_edges) - 1

        for i in range(n_ebins):
            bkg_map.plot(energy_bin_id=i)
            pyplot.savefig(f"{base_path}_map_bin{i}.png")
            pyplot.close(pyplot.gcf())

            if result is not None:
                fig, _ = bkg_map.plot_fit_check(result, energy_bin_id=i)
                fig.savefig(f"{base_path}_fit_check_bin{i}.png")
                pyplot.close(fig)

        if result is not None:
            fig, _ = bkg_map.plot_fit_spectrum(result)
            fig.savefig(f"{base_path}_fit_spectrum.png")
            pyplot.close(fig)

    @staticmethod
    def write_maps(bkg_maps, overwrite) -> None:
        """
        This method writes the generated bkgmaps to the corresponding output
        path. The output follows the definition in
        https://gamma-astro-data-formats.readthedocs.io/

        Prior to writing, a camera_response() model (see
        pybkgmodel.camera.CameraImage.fit_camera_response()) is fit to
        each map's observed counts, and the resulting smooth,
        Poisson-noise-free rate is what gets written out as the BKG
        column. If the fit does not converge, the map falls back to
        the raw counts-based rate (CameraImage.differential_rate()) and
        a warning is printed.

        Diagnostic plot images (the rate map, and, if the fit
        succeeded, the fit-quality checks) are saved alongside each
        output FITS file, see save_diagnostic_plots().

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
            fit_result = None
            # L-BFGS-B is fast but can struggle when the fit parameters
            # span very different scales; retry with the more robust
            # (but slower) gradient-free Nelder-Mead before giving up.
            for method, options in (
                ('L-BFGS-B', None),
                ('Nelder-Mead', dict(maxiter=50000, maxfev=50000, adaptive=True)),
            ):
                try:
                    dnde, fit_result = bkg_map.fitted_differential_rate(
                        p0, bounds=bounds, method=method, options=options
                    )
                    if fit_result.success:
                        break
                    dnde = None
                    fit_result = None
                except Exception:  # pylint: disable=broad-except
                    dnde = None
                    fit_result = None

            if dnde is None:
                message(
                    f"camera_response fit did not converge for '{key}'; "
                    "falling back to the raw counts-based background rate."
                )

            base_path, _ = os.path.splitext(key)
            BkgMakerBase.save_diagnostic_plots(bkg_map, base_path, result=fit_result)

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

        self.generate_runwise_maps()
        stacked_map = self.stack_maps(self.bkg_maps,
                                      self.x_edges,
                                      self.y_edges,
                                      self.e_edges
                                      )

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
