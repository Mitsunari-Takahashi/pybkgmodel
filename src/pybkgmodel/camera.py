import numpy as np
import scipy.special
import scipy.optimize

import astropy.units as u
import astropy.io.fits as pyfits
from astropy.wcs import WCS

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot


def king_function(x, y, sigma, gamma, ecc=0.0, phi=0.0, x0=0.0, y0=0.0):
    """
    2D King function (a generalization of a 2D Gaussian), commonly used
    to model the point spread function shape in the camera plane.
    In the gamma -> infinity limit this reduces to a 2D Gaussian with
    standard deviation sigma, elongated into an ellipse of eccentricity
    ecc, whose major axis is rotated by the angle phi with respect to
    the x axis.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates.
    sigma: array_like
        Width parameter of the King function, i.e. the geometric mean
        of the widths along the two ellipse axes.
    gamma: array_like
        Tail parameter of the King function.
    ecc: array_like
        Eccentricity (flattening) of the King function, 0 <= ecc < 1.
        ecc = 0 corresponds to the circularly symmetric King function.
    phi: array_like
        Rotation angle (in radians) of the ellipse major axis with
        respect to the x axis. Defaults to 0.
    x0, y0: array_like
        Center of the King function. Defaults to 0.

    Returns
    -------
    val: array_like
        King function value.
    """
    dx = x - x0
    dy = y - y0

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    x_rot = dx*cphi + dy*sphi
    y_rot = -dx*sphi + dy*cphi

    sigma_x = sigma / np.sqrt(1 - ecc)
    sigma_y = sigma * np.sqrt(1 - ecc)

    r2 = (x_rot/sigma_x)**2 + (y_rot/sigma_y)**2

    return (1 - 1/gamma) / (2*np.pi*sigma_x*sigma_y) * np.power(1 + r2/(2*gamma), -gamma)


def linear(y, a, b=1.0):
    """
    Linear function of y.

    b defaults to 1 because, inside camera_response(), the overall
    normalization of linear(y, a, b) * norm is degenerate between b and
    norm (b, norm) -> (b/k, norm*k) leaves the product unchanged. Fixing
    b = 1 removes that redundant degree of freedom, leaving norm as the
    sole overall normalization and a as the (now identifiable) relative
    slope.

    Parameters
    ----------
    y: array_like
        Independent variable.
    a: array_like
        Slope.
    b: array_like
        Intercept. Defaults to 1.

    Returns
    -------
    val: array_like
        a*y + b
    """
    return a*y + b


def log_parabola(e, norm, e0, alpha, beta):
    """
    Log-parabola function of energy.

    Parameters
    ----------
    e: array_like
        Energy.
    norm: array_like
        Normalization, i.e. the value at e = e0.
    e0: array_like
        Reference energy.
    alpha: array_like
        Spectral index at e = e0.
    beta: array_like
        Curvature parameter.

    Returns
    -------
    val: array_like
        norm * (e/e0)**(-alpha - beta*log(e/e0))
    """
    return norm * np.power(e/e0, -alpha - beta*np.log(e/e0))


def low_energy_cutoff(e, eth, s):
    """
    Exponential cutoff function that suppresses the value at low energies,
    i.e. it goes to 0 as e -> 0 and to 1 as e -> infinity.

    Parameters
    ----------
    e: array_like
        Energy.
    eth: array_like
        Threshold energy scale of the cutoff.
    s: array_like
        Steepness of the cutoff.

    Returns
    -------
    val: array_like
        exp(-(eth/e)**s)
    """
    return np.exp(-np.power(eth/e, s))


def camera_response(x, y, e, sigma0, delta_sigma, gamma, ecc, phi, a0, delta_a, norm, e0, alpha, beta, eth, s):
    """
    Camera background response model, defined as the product of:
    1. a 2D King function of (x, y),
    2. a linear function of y,
    3. a log-parabola function of e,
    4. a low-energy exponential cutoff function of e.

    The linear function of y is used with its intercept fixed to 1
    (see linear()), so that norm alone carries the overall
    normalization of the model; this removes the degeneracy that would
    otherwise exist between norm and the linear function's intercept.

    x, y and e are not fully separable: the King function's width and
    the linear function's slope are each allowed to vary with energy
    through a power law,

        sigma(e) = sigma0 * (e/e0)**delta_sigma
        a(e)     = a0     * (e/e0)**delta_a

    which is the lowest-order (linear in log(e/e0)) energy dependence
    compatible with sigma(e) > 0 at all energies. delta_sigma = 0 and
    delta_a = 0 recover the fully separable model.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates.
    e: array_like
        Energy.
    sigma0, delta_sigma, gamma, ecc, phi: array_like
        Parameters of the 2D King function, see king_function().
        sigma0 is the King function width at e = e0, and delta_sigma
        its power-law energy dependence (see above).
    a0, delta_a: array_like
        Parameters of the linear function of y, see linear().
        a0 is the slope at e = e0, and delta_a its power-law energy
        dependence (see above).
    norm, e0, alpha, beta: array_like
        Parameters of the log-parabola function of e, see log_parabola().
    eth, s: array_like
        Parameters of the low-energy cutoff function of e, see low_energy_cutoff().

    Returns
    -------
    val: array_like
        Camera response value.
    """
    sigma = sigma0 * np.power(e/e0, delta_sigma)
    a = a0 * np.power(e/e0, delta_a)

    return (
        king_function(x, y, sigma, gamma, ecc=ecc, phi=phi)
        * linear(y, a)
        * log_parabola(e, norm, e0, alpha, beta)
        * low_energy_cutoff(e, eth, s)
    )


def solid_angle_lat_lon_rectangle(theta_E, theta_W, phi_N, phi_S):
    """
    Calculate the solid angle of a latitude-longitude rectangle on a globe.
    Source of the formula: https://en.wikipedia.org/wiki/Solid_angle

    Parameters
    ----------
    phi_N: astropy.units.Quantity or astropy.coordinates.Angle
        North line of latitude
    phi_S: astropy.units.Quantity or astropy.coordinates.Angle
        South line of latitude
    theta_E: astropy.units.Quantity or astropy.coordinates.Angle
        East line of longitude
    theta_W: astropy.units.Quantity or astropy.coordinates.Angle
        West line of longitude

    Returns
    -------
    solid_angle: astropy.units.sr
        Calculated solid angle of a latitude-longitude rectangle.
    """

    phi_N = phi_N.to(u.Unit("rad"))
    phi_S = phi_S.to(u.Unit("rad"))
    theta_E = theta_E.to(u.Unit("rad"))
    theta_W = theta_W.to(u.Unit("rad"))
    solid_angle = (np.sin(phi_N) - np.sin(phi_S)) * (theta_E.to_value() - theta_W.to_value()) * u.sr

    return solid_angle


def cstat(y, model_y):
    """
    Poissonian C-statistics value.

    Parameters
    ----------
    y: array_like
        Measured counts. Must be integer.
    model_y: array_like
        Predicted counts

    Returns
    -------
    val: float
        2 * log-likelihood value.

    """
    val = -2 * np.sum(y * np.log(model_y) - model_y - scipy.special.gammaln(y+1))

    return val


def pwl2counts(emin, emax, norm, e0, index):
    """
    Integrated power law spectrum.

    Parameters
    ----------
    emin: array_like
        Minimal energy for integration.
    emax: array_like
        Minimal energy for integration.
    norm: array_like
        Spectral normalization.
    e0: array_like
        Spectrum normalization energy.
    index: array_like
        Spectral index

    Returns
    -------
    counts: array_like
        Integrated spectrum value in the range [emin; emax]

    """
    counts = norm * e0 / (index + 1) * ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))
    return counts


def nodespec_integral(energy_edges, dnde):
    """
    Differential node spectrum, integrated within the energy_edges.
    Nodes of the spectrum are assumed to be located
    at sqrt(energy_edges[1:] * energy_edges[:-1]).

    Parameters
    ----------
    energy_edges: array_like of astropy.units.Quantity
        Energy edges of the node spectrum bins.
    dnde: array_like of astropy.units.Quantity
        Differential flux values of the spectrum nodes.

    Returns
    -------
    counts: astropy.units.Quantity
        Integrated spectrum in each of the bins,
        defined by energy_edges.

    """

    if isinstance(energy_edges.unit, u.DexUnit):
        energy_edges = energy_edges.physical

    if isinstance(dnde.unit, u.DexUnit):
        dnde = dnde.physical

    energy = np.sqrt(energy_edges[1:] * energy_edges[:-1])
    counts = np.zeros(len(energy)) * u.one

    xunit = u.DexUnit(energy_edges.unit)
    yunit = u.DexUnit(dnde.unit)

    dx = np.diff(energy.to(xunit).value)
    dy = np.diff(dnde.to(yunit).value)
    indicies = dy / dx
    indicies = np.concatenate(
        (indicies[:1], indicies, indicies[-1:])
    )

    counts += pwl2counts(
        emin=energy_edges[:-1],
        emax=energy,
        norm=dnde,
        e0=energy,
        index=indicies[:-1]
    )
    counts += pwl2counts(
        emin=energy,
        emax=energy_edges[1:],
        norm=dnde,
        e0=energy,
        index=indicies[1:]
    )

    return counts


def node_cnt_diff(dnde, energy_edges, counts, poisson=False):
    """
    Summed squared difference between the node spectrum
    integral flux and the specified value.
    """
    ncounts = nodespec_integral(energy_edges, dnde)

    if not poisson:
        delta = (counts - ncounts)**2
    else:
        delta = cstat(counts, ncounts)

    return delta.sum()


class CameraImage:
    def __init__(self, counts, xedges, yedges, energy_edges, center=None, mask=None, exposure=None):
        nx = xedges.size - 1
        ny = yedges.size - 1

        if mask is None:
            mask = np.ones((nx, ny), dtype=bool)

        if exposure is None:
            exposure = np.ones((nx, ny), dtype=np.float) * u.s
        elif exposure.shape == ():
            exposure = np.repeat(exposure, nx * ny).reshape((nx, ny))

        self.raw_counts = counts
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center
        self.mask = mask
        self.raw_exposure = exposure

        self.pixel_coords = self.get_pixel_coords()
        self.pixel_area = self.get_pixel_areas()

    @classmethod
    def from_events(cls, event_file, xedges, yedges, energy_edges):
        center = cls.get_poiting(event_file)
        image = cls.bin_events(event_file, xedges, yedges, energy_edges)

        return cls(image, xedges, yedges, energy_edges, center=center, exposure=event_file.events.eff_obs_time)

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Center':.<20s}: {self.center}
    {'X range':.<20s}: [{self.xedges.min():.1f}, {self.xedges.max():.1f}]
    {'Y range':.<20s}: [{self.yedges.min():.1f}, {self.yedges.max():.1f}]
    {'X bins':.<20s}: {len(self.xedges) - 1}
    {'X bins':.<20s}: {len(self.yedges) - 1}
    {'Exposure (mean)':.<20s}: {self.raw_exposure[self.mask].mean()}
"""
        )

        return super().__repr__()

    @classmethod
    def bin_events(cls, event_file, xedges, yedges, energy_edges):
        pass

    def get_pixel_coords(self):
        pass

    def get_pixel_areas(self):
        pass

    @classmethod
    def get_poiting(cls, event_file):
        return SkyCoord(ra=event_file.pointing_ra.mean(), dec=event_file.pointing_dec.mean())

    @property
    def counts(self):
        return self.raw_counts * self.mask

    @property
    def exposure(self):
        return self.raw_exposure * self.mask

    @property
    def rate(self):
        return self.counts / self.raw_exposure / self.pixel_area

    def differential_rate(self, index=None):
        """
        Differential count rate assuming the power law
        spectral shape dN/dE = A*(E/E0)**index with the specified
        spectral index. Rate is calculated in at e0 = (emin * emax)**0.5
        following the existing energy binning.

        Parameters
        ----------
        index: float
            Power law spectral index to assume.
            If none, will be dynamically determined assuming
            a "node function" for the spectral shape.

        Returns
        -------
        differential_rate: array_like astropy.unit.Quantity
            Computed rate of the same shape as the camera image.
        """

        emin = self.energy_edges[:-1]
        emax = self.energy_edges[1:]
        e0 = (emin * emax)**0.5

        if index is None:
            # Approximate solution
            index = -2
            int2diff = (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))

            dnde = self.counts * int2diff[:, None, None]

            # Final value
            dnde_unit = u.DexUnit(dnde.unit)
            for xi in range(self.rate.shape[1]):
                for yi in range(self.rate.shape[2]):
                    if not np.any(dnde[:, xi, yi] == 0):
                        opt = scipy.optimize.minimize(
                            lambda x: node_cnt_diff((x*dnde_unit).physical, self.energy_edges, self.counts[:, xi, yi], poisson=True),
                            x0=dnde[:, xi, yi].to(dnde_unit).value
                        )

                        if opt.success == True:
                            dnde[:, xi, yi] = (opt.x * dnde_unit).physical

            dnde = dnde / self.raw_exposure / self.pixel_area

        else:
            int2diff = (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))

            dnde = self.rate * int2diff[:, None, None]

        return dnde

    def fitted_differential_rate(self, p0, bounds=None, index=-2, method='L-BFGS-B', **kwargs):
        """
        Same as differential_rate(index=index), but based on a smooth
        camera_response() fit to the observed counts (see
        fit_camera_response()) instead of on the observed counts
        themselves. Produces a statistically smoothed background rate
        map, free of the Poisson noise of the raw counts.

        Parameters
        ----------
        p0: array_like
            Initial guess for the camera_response() fit parameters,
            see fit_camera_response().
        bounds: sequence of (min, max), optional
            Bounds on the fit parameters, see fit_camera_response().
        index: float
            Power law spectral index used to convert the integrated,
            per-bin model counts into a differential rate, see
            differential_rate(). Defaults to -2.
        method: str
            Optimization method, see fit_camera_response().
        **kwargs:
            Any additional keyword arguments are passed to
            fit_camera_response().

        Returns
        -------
        dnde: array_like astropy.unit.Quantity
            Fitted differential rate, of the same shape as the camera
            image, see differential_rate().
        result: scipy.optimize.OptimizeResult
            Result of the underlying fit_camera_response() call.
        """
        result = self.fit_camera_response(p0, bounds=bounds, method=method, **kwargs)

        emin = self.energy_edges[:-1]
        emax = self.energy_edges[1:]
        e0 = (emin * emax)**0.5
        int2diff = (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))

        rate = result.model_counts / self.raw_exposure / self.pixel_area
        dnde = rate * int2diff[:, None, None]

        return dnde, result

    def fit_camera_response(self, p0, bounds=None, method='L-BFGS-B', **kwargs):
        """
        Fit the product of camera_response() and the exposure map to the
        observed counts, assuming the counts follow Poisson statistics
        (i.e. minimizing the C-stat, see cstat()).

        Only the unmasked pixels (see mask, mask_half(), mask_region())
        are included in the fit.

        Parameters
        ----------
        p0: array_like
            Initial guess for the camera_response() fit parameters
            (sigma0, delta_sigma, gamma, ecc, phi, a0, delta_a, norm, e0,
            alpha, beta, eth, s).
            x, y and e (camera coordinates and energy) are not fitted:
            they are set to the pixel / energy bin centers of this image,
            in degrees and TeV respectively.
        bounds: sequence of (min, max), optional
            Bounds on the fit parameters, passed to scipy.optimize.minimize.
        method: str
            Optimization method, passed to scipy.optimize.minimize.
            Defaults to 'L-BFGS-B', which supports bounds. Note that the
            fit parameters can differ by orders of magnitude in scale
            (e.g. norm vs. ecc), which can make gradient-based methods
            converge poorly with their default numerical-differentiation
            step size; passing scipy's `finite_diff_rel_step` keyword
            argument, rescaling p0/bounds, or using a gradient-free
            method (e.g. 'Nelder-Mead') can help in that case.
        **kwargs:
            Any additional keyword arguments are passed to
            scipy.optimize.minimize.

        Returns
        -------
        result: scipy.optimize.OptimizeResult
            Result of the fit. result.x contains the best fit parameters,
            in the same order as p0. result.model_counts is the fitted
            camera_response(), multiplied by the exposure, evaluated on
            the pixel / energy bin centers of this image: a smooth count
            map of the same shape as CameraImage.counts.
        """
        x = ((self.xedges[1:] + self.xedges[:-1]) / 2).to_value(u.deg)
        y = ((self.yedges[1:] + self.yedges[:-1]) / 2).to_value(u.deg)
        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(u.TeV)

        ee, xx, yy = np.meshgrid(e, x, y, indexing='ij')

        exposure = self.raw_exposure.to_value(u.s)
        counts = self.raw_counts
        mask = np.broadcast_to(self.mask, counts.shape)

        def neg_log_likelihood(params):
            model_counts = camera_response(xx, yy, ee, *params) * exposure
            return cstat(counts[mask], model_counts[mask])

        result = scipy.optimize.minimize(
            neg_log_likelihood,
            x0=p0,
            bounds=bounds,
            method=method,
            **kwargs
        )

        result.model_counts = camera_response(xx, yy, ee, *result.x) * exposure

        return result

    def plot_fit_check(self, result, energy_bin_id=0, ax_unit='deg', cmap='viridis'):
        """
        Plot a spatial comparison between the observed counts and the
        counts predicted by a camera_response() fit (see
        fit_camera_response()), for one energy bin, as a visual check
        of the fit quality.

        Produces three panels: the observed counts, the fitted model
        counts and the residuals, expressed as Poisson-equivalent
        Gaussian pulls (data - model) / sqrt(model). Masked pixels
        (see mask, mask_half(), mask_region()) are left blank.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts.
        energy_bin_id: int
            Energy bin to plot.
        ax_unit: str
            Unit to use for the x/y axes.
        cmap: str
            Colormap to use for the data/model count maps.

        Returns
        -------
        fig, axes: matplotlib figure and array of 3 axes
            (data, model, pull).
        """
        mask = self.mask
        data = self.counts[energy_bin_id]
        model = result.model_counts[energy_bin_id] * mask

        pull = np.full(data.shape, np.nan)
        pull[mask] = (data[mask] - model[mask]) / np.sqrt(model[mask])

        xedges = self.xedges.to_value(ax_unit)
        yedges = self.yedges.to_value(ax_unit)

        vmax = max(data.max(), model.max())
        pmax = np.nanmax(np.abs(pull)) if np.any(mask) else 1

        fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))

        for ax, val, title, kwargs in zip(
            axes,
            (data, model, pull),
            ('Data counts', 'Model counts', 'Pull: (data - model) / sqrt(model)'),
            (
                dict(vmin=0, vmax=vmax, cmap=cmap),
                dict(vmin=0, vmax=vmax, cmap=cmap),
                dict(vmin=-pmax, vmax=pmax, cmap='coolwarm')
            )
        ):
            im = ax.pcolormesh(xedges, yedges, val.transpose(), **kwargs)
            ax.set_title(title)
            ax.set_xlabel(f'X [{ax_unit}]')
            ax.set_ylabel(f'Y [{ax_unit}]')
            fig.colorbar(im, ax=ax)

        emin = self.energy_edges[energy_bin_id]
        emax = self.energy_edges[energy_bin_id + 1]
        fig.suptitle(f'Energy bin {energy_bin_id}: [{emin:.2f}, {emax:.2f}]')
        fig.tight_layout()

        return fig, axes

    def plot_fit_spectrum(self, result, e_unit='TeV'):
        """
        Plot the observed vs. fitted (model) counts, summed over all
        unmasked pixels, as a function of energy, together with the
        residuals (pulls), as a check of the fit quality across the
        whole energy range.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts.
        e_unit: str
            Unit to use for the energy axis.

        Returns
        -------
        fig, (ax_spec, ax_pull): matplotlib figure and axes
            (count spectrum, pull).
        """
        data = self.counts.sum(axis=(1, 2))
        model = (result.model_counts * self.mask).sum(axis=(1, 2))

        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(e_unit)
        data_err = np.sqrt(data)
        pull = (data - model) / np.sqrt(model)

        fig, (ax_spec, ax_pull) = pyplot.subplots(
            2, 1, sharex=True, figsize=(6, 6),
            gridspec_kw=dict(height_ratios=[3, 1])
        )

        ax_spec.errorbar(e, data, yerr=data_err, fmt='o', color='k', label='Data')
        ax_spec.plot(e, model, '-', color='C1', label='Model')
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')
        ax_spec.set_ylabel('Counts')
        ax_spec.legend()

        ax_pull.axhline(0, color='gray', ls='--')
        ax_pull.plot(e, pull, 'o', color='k')
        ax_pull.set_xlabel(f'Energy [{e_unit}]')
        ax_pull.set_ylabel('Pull')

        fig.tight_layout()

        return fig, (ax_spec, ax_pull)

    def mask_half(self, pointer):
        """Excludes the half of the camera containing the sources.

        Parameters
        ----------
        pointer : SkyCoord
            Source position in the camera coordinate system.
        """
        offset_delta = Angle('90d')

        pixel_position_angles = self.center.position_angle(self.pixel_coords)
        pointer_position_angle = self.center.position_angle(pointer)
        position_angle_offest = (pixel_position_angles - pointer_position_angle).wrap_at('180d')

        to_mask = (position_angle_offest >= -offset_delta) & (position_angle_offest < offset_delta)

        self.mask[to_mask] = False

    def mask_region(self, region):
        """Masks (empties) all camera pixel contained in the provided sky region.

        Parameters
        ----------
        region : region object
            See https://astropy-regions.readthedocs.io/en/stable/contains.html
        """
        dummy_wcs = WCS(naxis=2)
        # Taken from https://docs.astropy.org/en/stable/wcs/example_create_imaging.html
        # Set up an "Airy's zenithal" projection
        # Vector properties may be set with Python lists, or np arrays
        dummy_wcs.wcs.crpix = [-234.75, 8.3393]
        dummy_wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
        dummy_wcs.wcs.crval = [0, -90]
        dummy_wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        dummy_wcs.wcs.set_pv([(2, 1, 45.0)])

        in_region = region.contains(self.pixel_coords, dummy_wcs)
        self.mask[in_region] = False

    def mask_reset(self):
        """_summary_
        """
        self.mask = np.ones((self.xedges.size - 1, self.yedges.size - 1), dtype=bool)


    def plot(self, energy_bin_id=0, ax_unit='deg', val_unit='1/s', **kwargs):
        pyplot.xlabel(f'X [{ax_unit}]')
        pyplot.ylabel(f'Y [{ax_unit}]')
        pyplot.pcolormesh(
            self.xedges.to(ax_unit).value,
            self.yedges.to(ax_unit).value,
            (self.counts[energy_bin_id] / self.raw_exposure).to_value(val_unit).transpose(),
            **kwargs
        )
        pyplot.colorbar(label=f'rate [{val_unit}]')


class RectangularCameraImage(CameraImage):
    @classmethod
    def bin_events(cls, event_file, xedges, yedges, energy_edges):
        center = cls.get_poiting(event_file)
        events = SkyCoord(ra=event_file.event_ra, dec=event_file.event_dec)

        cam = events.transform_to(center.skyoffset_frame())

        hist, _ = np.histogramdd(
            sample=(
                event_file.event_energy,
                cam.lon,
                cam.lat
            ),
            bins=(
                energy_edges,
                xedges,
                yedges
            )
        )

        return hist

    def get_pixel_coords(self):
        if self.center is None:
            frame=None
        else:
            frame=self.center.skyoffset_frame()

        x = (self.xedges[1:] + self.xedges[:-1]) / 2
        y = (self.yedges[1:] + self.yedges[:-1]) / 2
        xx, yy = np.meshgrid(x, y, indexing='ij')

        pixel_coords = SkyCoord(
            xx,
            yy,
            frame=frame
        )

        return pixel_coords

    def get_pixel_areas(self):
        nx = self.xedges.size - 1
        ny = self.yedges.size - 1

        area = np.zeros((nx, ny)) * u.sr

        for i in range(nx):
            for j in range(ny):
                area[i, j] = solid_angle_lat_lon_rectangle(self.xedges[i], self.xedges[i+1], self.yedges[j], self.yedges[j+1])

        return area

    def to_hdu(self, name='BACKGROUND', bkg_rate=None):
        """
        Parameters
        ----------
        name: str
            Name of the output HDU.
        bkg_rate: array_like astropy.unit.Quantity, optional
            Differential background rate to write out, of the same
            shape as differential_rate(). If None (default), it is
            computed on the fly as differential_rate(index=-2), i.e.
            directly from the (noisy) observed counts. Pass e.g. the
            dnde array returned by fitted_differential_rate() to write
            out a smooth, fit-based background model instead.

        Returns
        -------
        hdu: astropy.io.fits.BinTableHDU
        """
        energ_lo = self.energy_edges[:-1]
        energ_hi = self.energy_edges[1:]

        detx_lo = self.xedges[:-1]
        detx_hi = self.xedges[1:]

        dety_lo = self.yedges[:-1]
        dety_hi = self.yedges[1:]

        if bkg_rate is None:
            bkg_rate = self.differential_rate(index=-2)

        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_detx_lo = pyfits.Column(name='DETX_LO', unit='deg', format=f'{detx_lo.size}E', array=[detx_lo])
        col_detx_hi = pyfits.Column(name='DETX_HI', unit='deg', format=f'{detx_hi.size}E', array=[detx_hi])
        col_dety_lo = pyfits.Column(name='DETY_LO', unit='deg', format=f'{dety_lo.size}E', array=[dety_lo])
        col_dety_hi = pyfits.Column(name='DETY_HI', unit='deg', format=f'{dety_hi.size}E', array=[dety_hi])

        col_bkg_rate = pyfits.Column(
            name='BKG',
            unit='s^-1 MeV^-1 sr^-1',
            format=f"{bkg_rate.size}E",
            array=[
                bkg_rate.to('1 / (s * MeV * sr)').value.transpose()
            ],
            dim=str(bkg_rate.shape))

        columns = [
            col_energ_lo,
            col_energ_hi,
            col_detx_lo,
            col_detx_hi,
            col_dety_lo,
            col_dety_hi,
            col_bkg_rate
        ]

        col_defs = pyfits.ColDefs(columns)
        hdu = pyfits.BinTableHDU.from_columns(col_defs)
        hdu.name = name

        hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        hdu.header['HDUVERS'] = '0.2'
        hdu.header['HDUCLASS'] = 'GADF'
        hdu.header['HDUCLAS1'] = 'RESPONSE'
        hdu.header['HDUCLAS2'] = 'BKG'
        hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu.header['HDUCLAS4'] = 'BKG_3D'

        return hdu
