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


def super_gaussian(x, y, sigma, p, ecc=0.0, phi=0.0, x0=0.0, y0=0.0):
    """
    2D super-Gaussian function, a generalization of a 2D Gaussian with
    a tunable flatness at the center, elongated into an ellipse of
    eccentricity ecc, whose major axis is rotated by the angle phi
    with respect to the x axis.

    p = 1 recovers an ordinary 2D Gaussian of standard deviation
    sigma. p > 1 flattens the peak into a plateau near the center
    (with a steeper falloff further out, unlike king_function()'s
    power-law tail, this one always falls off exponentially); p < 1
    gives a sharper (more cusped) peak than a Gaussian.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates.
    sigma: array_like
        Width parameter, i.e. the geometric mean of the widths along
        the two ellipse axes.
    p: array_like
        Flatness exponent, p > 0. p = 1 is a standard Gaussian; p > 1
        flattens the center into a plateau.
    ecc: array_like
        Eccentricity (flattening), 0 <= ecc < 1. ecc = 0 corresponds
        to the circularly symmetric function. Defaults to 0.
    phi: array_like
        Rotation angle (in radians) of the ellipse major axis with
        respect to the x axis. Defaults to 0.
    x0, y0: array_like
        Center. Defaults to 0.

    Returns
    -------
    val: array_like
        Super-Gaussian function value.
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

    # Exact 2D normalization for the elliptical super-Gaussian; reduces
    # to the usual 1/(2*pi*sigma_x*sigma_y) Gaussian normalization at
    # p = 1 (scipy.special.gamma(1) = 1).
    prefactor = p / (2*np.pi*sigma_x*sigma_y*scipy.special.gamma(1/p))

    return prefactor * np.exp(-np.power(r2/2, p))


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


def camera_response(x, y, e,
                     sigma_core0, delta_sigma_core, x0_core, y0_core,
                     sigma_tail, gamma_tail, ecc_tail0, delta_ecc_tail, phi_tail,
                     w0, delta_w,
                     norm, e0, alpha, beta):
    """
    Camera background response model, defined as the product of:
    1. a spatial (x, y) distribution, itself the sum of two components:
       a. a circularly symmetric 2D Gaussian "core", and
       b. an elliptical 2D King function "tail" (see king_function()),
       mixed as w*core + (1-w)*tail, with each individually normalized
       to integrate to 1 over the camera plane, so w is the fraction
       of the spatial distribution's total (normalized) weight
       contributed by the core (see below for its energy dependence).
    2. a linear function of y (currently fixed to a constant 1, see
       linear() -- both its slope and intercept are fixed for now, as
       this was found to make the fit more stable),
    3. a log-parabola function of e.

    The core is centered at (x0_core, y0_core), a free fit parameter;
    the tail is fixed at the camera center (0, 0). The core's width
    varies with energy through a power law,

        sigma_core(e) = sigma_core0 * (e/e0)**delta_sigma_core

    which is the lowest-order (linear in log(e/e0)) energy dependence
    compatible with sigma_core(e) > 0 at all energies. delta_sigma_core
    = 0 recovers an energy-independent core width. The core's center
    (x0_core, y0_core) itself has no energy dependence.

    The tail's width (sigma_tail) has no energy dependence, but its
    eccentricity does, through a logistic (sigmoid) function of
    log(e/e0),

        ecc_tail(e) = sigmoid(logit(ecc_tail0) + delta_ecc_tail*log(e/e0))

    where logit(p) = log(p/(1-p)) and sigmoid = logit's inverse. This
    is the eccentricity analog of the sigma_core(e) power law above:
    the lowest-order (linear in log(e/e0)) energy dependence that
    still keeps ecc_tail(e) within its valid (0, 1) range at all
    energies (a plain linear or power-law term could not, since
    eccentricity is bounded on both sides). ecc_tail(e0) = ecc_tail0
    exactly, and delta_ecc_tail = 0 recovers an energy-independent
    tail eccentricity.

    The core fraction w is bounded the same way (0, 1) and uses the
    same logistic construction,

        w(e) = sigmoid(logit(w0) + delta_w*log(e/e0))

    so w(e0) = w0 exactly, and delta_w = 0 recovers an energy-independent
    core/tail mix.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates.
    e: array_like
        Energy.
    sigma_core0, delta_sigma_core, x0_core, y0_core: array_like
        Parameters of the Gaussian core (a super_gaussian() with
        p = 1, ecc = 0, phi = 0). sigma_core0 is its width at e = e0,
        and delta_sigma_core its power-law energy dependence (see
        above); x0_core, y0_core its (energy-independent) center.
    sigma_tail, gamma_tail, ecc_tail0, delta_ecc_tail, phi_tail: array_like
        Parameters of the King function tail, see king_function(),
        fixed at the camera center (0, 0). ecc_tail0 is its
        eccentricity at e = e0, and delta_ecc_tail its logistic energy
        dependence (see above).
    w0, delta_w: array_like
        Fraction (0 < w < 1) of the spatial distribution's weight in
        the core, vs. (1 - w) in the tail (see above). w0 is its value
        at e = e0, and delta_w its logistic energy dependence (see
        above).
    norm, e0, alpha, beta: array_like
        Parameters of the log-parabola function of e, see log_parabola().

    Returns
    -------
    val: array_like
        Camera response value.
    """
    sigma_core = sigma_core0 * np.power(e/e0, delta_sigma_core)

    ecc_tail_logit0 = np.log(ecc_tail0 / (1 - ecc_tail0))
    ecc_tail = 1 / (1 + np.exp(-(ecc_tail_logit0 + delta_ecc_tail*np.log(e/e0))))

    w_logit0 = np.log(w0 / (1 - w0))
    w = 1 / (1 + np.exp(-(w_logit0 + delta_w*np.log(e/e0))))

    core = super_gaussian(x, y, sigma_core, p=1.0, x0=x0_core, y0=y0_core)
    tail = king_function(x, y, sigma_tail, gamma_tail, ecc=ecc_tail, phi=phi_tail)
    spatial = w*core + (1 - w)*tail

    return (
        spatial
        * linear(y, a=0.0)
        * log_parabola(e, norm, e0, alpha, beta)
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

        # Using model_rate (exposure-independent) directly, rather than
        # model_counts / exposure, avoids a 0/0 division in pixels with
        # zero exposure (e.g. never covered by any stacked run/always
        # excluded); model_rate is mathematically identical to
        # model_counts / exposure wherever exposure is nonzero.
        rate = result.model_rate / self.pixel_area
        dnde = rate * int2diff[:, None, None]

        return dnde, result

    def fit_camera_response(self, p0, bounds=None, method='L-BFGS-B', **kwargs):
        """
        Fit the product of camera_response() and the exposure map to the
        observed counts, assuming the counts follow Poisson statistics
        (i.e. minimizing the C-stat, see cstat()).

        Only the unmasked pixels (see mask, mask_half(), mask_region())
        are included in the fit. In energy, only the bin with the
        highest total count (in the unmasked pixels) and all bins
        above it are included; bins below the peak are excluded from
        the fit entirely.

        Parameters
        ----------
        p0: array_like
            Initial guess for the camera_response() fit parameters
            (sigma_core0, delta_sigma_core, x0_core, y0_core,
            sigma_tail, gamma_tail, ecc_tail0, delta_ecc_tail, phi_tail,
            w0, delta_w, norm, e0, alpha, beta).
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

        # Also restrict the fit to the energy bin with the highest
        # total count (in the unmasked pixels) and all bins above it:
        # bins below the peak carry little information to constrain
        # the energy-dependent fit parameters and would otherwise just
        # add spurious -model_counts terms to the fit statistic.
        bin_totals = (counts * mask).sum(axis=(1, 2))
        peak_bin = np.argmax(bin_totals)
        in_range = np.arange(len(bin_totals)) >= peak_bin
        mask = mask & in_range[:, None, None]

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
        # Exposure-independent fitted rate (model_counts = model_rate *
        # exposure): unlike model_counts, this stays well-defined even
        # in pixels with zero exposure, and is what fitted_differential_rate()
        # / posterior_differential_rate() fall back to there instead of
        # dividing by zero.
        result.model_rate = camera_response(xx, yy, ee, *result.x) / u.s

        return result

    def posterior_counts(self, result, prior_strength=1.0):
        """
        Bayesian per-bin update of the observed counts, using the
        camera_response() fit (see fit_camera_response()) as the prior
        and the observed counts as the data.

        Each bin's true mean count mu is given a Gamma(k, k/lambda)
        prior, where lambda = result.model_counts is the fitted,
        statistically smooth model (the prior mean) and k =
        prior_strength is the prior's weight, expressed as an
        equivalent number of prior "pseudo-counts". The observed count
        n ~ Poisson(mu) then updates this to the conjugate posterior
        Gamma(k + n, k/lambda + 1), whose mean is

            posterior_mean = (k + n) * lambda / (k + lambda)

        This shrinks noisy, low-count bins toward the smooth fitted
        model (posterior_mean -> lambda as n, lambda << k) while
        letting well-measured, high-count bins be dominated by the
        data (posterior_mean -> n as lambda, n >> k) -- a smooth,
        Poisson-noise-reduced count map that still tracks genuine
        small-scale features in the data the smooth fit cannot
        capture.

        Masked pixels (see mask, mask_half(), mask_region()) have no
        valid data to update the prior with -- e.g. a source region's
        raw counts are contaminated by real signal, not pure
        background -- so no likelihood term is evaluated there at all,
        and the posterior is left equal to the prior (lambda)
        unchanged. Note that replacing the *data* with 0 in that case
        (rather than omitting the update) would not avoid this
        contamination cleanly either: it would feed the update a false
        "0 counts observed" likelihood, biasing the posterior below
        lambda instead.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts, used as
            the per-bin prior mean.
        prior_strength: float
            Equivalent number of prior "pseudo-counts" k (k > 0)
            controlling how strongly the fitted model constrains the
            posterior relative to the data. Larger values weight the
            prior (the fit) more heavily; smaller values let the raw
            data dominate. Defaults to 1.0.

        Returns
        -------
        counts: array_like
            Posterior mean count in each bin, of the same shape as
            CameraImage.counts.
        """
        n = self.raw_counts
        lam = result.model_counts
        k = prior_strength

        updated = (k + n) * lam / (k + lam)
        mask = np.broadcast_to(self.mask, n.shape)

        return np.where(mask, updated, lam)

    def posterior_rate(self, result, prior_strength=1.0):
        """
        Bayesian posterior counts (see posterior_counts()), converted
        to a plain per-pixel rate (posterior_counts / exposure, in
        1/s) -- the same rate quantity plot() shows for the raw data.

        Pixels with zero exposure (e.g. never covered by any stacked
        run/always excluded, see mask, mask_half(), mask_region())
        give a 0/0 division here; there, this falls back to the
        exposure-independent fitted rate (result.model_rate, see
        fit_camera_response()) instead of propagating a NaN.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), used as the prior, see
            posterior_counts().
        prior_strength: float
            Prior weight, see posterior_counts().

        Returns
        -------
        rate: array_like astropy.unit.Quantity
            Posterior rate, of the same shape as CameraImage.counts.
        """
        posterior_counts = self.posterior_counts(result, prior_strength=prior_strength)

        with np.errstate(invalid='ignore', divide='ignore'):
            rate = posterior_counts / self.raw_exposure

        invalid = ~np.isfinite(rate.value)
        if np.any(invalid):
            rate = rate.copy()
            rate[invalid] = result.model_rate[invalid]

        return rate

    def posterior_differential_rate(self, result, prior_strength=1.0, index=-2):
        """
        Same as fitted_differential_rate(), but based on the Bayesian
        per-bin posterior counts (see posterior_counts()) instead of
        directly on the camera_response() fit's model counts.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), used as the prior, see
            posterior_counts().
        prior_strength: float
            Prior weight, see posterior_counts().
        index: float
            Power law spectral index used to convert the integrated,
            per-bin posterior counts into a differential rate, see
            differential_rate(). Defaults to -2.

        Returns
        -------
        dnde: array_like astropy.unit.Quantity
            Posterior differential rate, of the same shape as the
            camera image, see differential_rate().
        """
        emin = self.energy_edges[:-1]
        emax = self.energy_edges[1:]
        e0 = (emin * emax)**0.5
        int2diff = (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))

        rate = self.posterior_rate(result, prior_strength=prior_strength) / self.pixel_area
        dnde = rate * int2diff[:, None, None]

        return dnde

    def plot_fit_check(self, result, energy_bin_id=0, ax_unit='deg', cmap='viridis', prior_strength=10.0, val_unit='1/s'):
        """
        Plot a spatial comparison between the observed rate and the
        Bayesian posterior rate (see posterior_rate()) -- the
        camera_response() fit (see fit_camera_response()) used as the
        prior, updated bin-by-bin against the observed data -- for one
        energy bin, as a visual check of the fit quality.

        Produces three panels: the observed rate, the posterior rate
        and the residuals, expressed as the relative residual
        (data - posterior) / posterior. Masked pixels (see mask,
        mask_half(), mask_region()) are left blank in the data and
        residual panels (there is no trustworthy data to show or check
        there), but the posterior panel shows the actual
        posterior_rate() value there too (the prior, unaffected by
        data) -- the same rate written out for that pixel by
        pybkgmodel.processing.BkgMakerBase.write_maps().

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
            Colormap to use for the data/posterior rate maps.
        prior_strength: float
            Prior weight passed to posterior_counts() / posterior_rate().
            Defaults to 10.0.
        val_unit: str
            Unit to use for the data/posterior rate maps.

        Returns
        -------
        fig, axes: matplotlib figure and array of 3 axes
            (data, posterior, pull).
        """
        mask = self.mask
        data_counts = self.counts[energy_bin_id]
        posterior_counts = self.posterior_counts(result, prior_strength=prior_strength)[energy_bin_id]

        pull = np.full(data_counts.shape, np.nan)
        pull[mask] = (data_counts[mask] - posterior_counts[mask]) / posterior_counts[mask]

        with np.errstate(invalid='ignore', divide='ignore'):
            data = (data_counts / self.raw_exposure).to_value(val_unit)
        model = self.posterior_rate(result, prior_strength=prior_strength)[energy_bin_id].to_value(val_unit)

        xedges = self.xedges.to_value(ax_unit)
        yedges = self.yedges.to_value(ax_unit)

        vmax = max(np.nanmax(data), model.max())
        pmax = np.nanmax(np.abs(pull)) if np.any(mask) else 1

        fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))

        for ax, val, title, kwargs in zip(
            axes,
            (data, model, pull),
            (f'Data rate [{val_unit}]', f'Posterior rate [{val_unit}]', 'Residual: (data - posterior) / posterior'),
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

    def plot_fit_spectrum(self, result, e_unit='TeV', prior_strength=10.0, val_unit='1/s'):
        """
        Plot the observed vs. Bayesian posterior rate (see
        posterior_rate() -- the camera_response() fit used as the
        prior, updated bin-by-bin against the observed data) as a
        function of energy, together with the residuals, as a check of
        the fit quality across the whole energy range.

        The plotted posterior curve is summed (per energy bin) over
        *all* pixels' rates (masked included), matching the total
        actually written out by
        pybkgmodel.processing.BkgMakerBase.write_maps() -- masked
        pixels contribute their prior (unaffected by data) rate. The
        data curve sums the observed rate over the unmasked pixels
        only (there is no trustworthy rate to show for a masked pixel,
        see posterior_rate()), with its error bars obtained by
        properly propagating the per-pixel Poisson counting variance.
        The residual, (data - posterior) / posterior, is computed in
        count space (summed over the unmasked pixels only); being a
        ratio, its value is the same whether computed from counts or
        from the corresponding rates.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts.
        e_unit: str
            Unit to use for the energy axis.
        prior_strength: float
            Prior weight passed to posterior_counts() / posterior_rate().
            Defaults to 10.0.
        val_unit: str
            Unit to use for the count-rate axis.

        Returns
        -------
        fig, (ax_spec, ax_pull): matplotlib figure and axes
            (rate spectrum, residual).
        """
        posterior_counts = self.posterior_counts(result, prior_strength=prior_strength)
        posterior_rate = self.posterior_rate(result, prior_strength=prior_strength)

        mask = np.broadcast_to(self.mask, self.raw_counts.shape)

        # Residual in count space, summed over the unmasked pixels only.
        data_counts = self.counts.sum(axis=(1, 2))
        model_counts_unmasked = (posterior_counts * self.mask).sum(axis=(1, 2))
        residual = (data_counts - model_counts_unmasked) / model_counts_unmasked

        # Displayed rate curves: per-pixel rate summed over energy,
        # with the Poisson counting variance propagated the same way
        # (Var(counts/exposure) = counts/exposure**2).
        with np.errstate(invalid='ignore', divide='ignore'):
            data_rate_map = self.raw_counts / self.raw_exposure
            data_var_map = self.raw_counts / self.raw_exposure**2

        data_rate_map = np.where(mask, data_rate_map, 0)
        data_var_map = np.where(mask, data_var_map, 0)

        data = data_rate_map.sum(axis=(1, 2)).to_value(val_unit)
        data_err = np.sqrt(data_var_map.sum(axis=(1, 2)).to_value(u.Unit(val_unit)**2))
        model = posterior_rate.sum(axis=(1, 2)).to_value(val_unit)

        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(e_unit)

        fig, (ax_spec, ax_pull) = pyplot.subplots(
            2, 1, sharex=True, figsize=(6, 6),
            gridspec_kw=dict(height_ratios=[3, 1])
        )

        ax_spec.errorbar(e, data, yerr=data_err, fmt='o', color='k', label='Data')
        ax_spec.plot(e, model, '-', color='C1', label='Posterior')
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')
        ax_spec.set_xlim(left=0.1)
        ax_spec.set_ylim(bottom=np.min(data[data > 0]) / 2, top=1.5 * data.max())
        ax_spec.set_ylabel(f'Rate [{val_unit}]')
        ax_spec.legend()

        ax_pull.axhline(0, color='gray', ls='--')
        ax_pull.plot(e, residual, 'o', color='k')
        ax_pull.set_xlabel(f'Energy [{e_unit}]')
        ax_pull.set_ylabel('Residual')
        ax_pull.set_ylim(-2, 2)

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
