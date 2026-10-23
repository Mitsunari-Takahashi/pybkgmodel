import numpy as np
import scipy.special
import scipy.optimize
import scipy.stats

import astropy.units as u
import astropy.io.fits as pyfits
from astropy.wcs import WCS

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def super_gaussian(x, y, sigma, p, ecc=0.0, phi=0.0, x0=0.0, y0=0.0, a_u=0.0, a_v=0.0):
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

    The function can be made asymmetric along both ellipse axes: on
    the positive side of the major axis (in the direction phi) the
    width along it is multiplied by (1 + a_u), on the negative side by
    (1 - a_u), and likewise along the minor axis with a_v. The value
    and its first derivatives stay continuous across the axes, and the
    normalization is unchanged, as the widths on both sides add up to
    twice the symmetric one.

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
    a_u, a_v: array_like
        Asymmetry along the major and minor ellipse axes, -1 < a < 1
        (see above). Defaults to 0 (symmetric).

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

    # Side-dependent widths of the asymmetric version (the
    # normalization below only depends on their two-sided averages,
    # sigma_x and sigma_y).
    side_x = sigma_x * np.where(x_rot >= 0, 1 + a_u, 1 - a_u)
    side_y = sigma_y * np.where(y_rot >= 0, 1 + a_v, 1 - a_v)

    r2 = (x_rot/side_x)**2 + (y_rot/side_y)**2

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


# Names of the camera_response() fit parameters, in the order expected
# by camera_response() and CameraImage.fit_camera_response(), and of
# result.x returned by the latter (see camera_response_params()).
CAMERA_RESPONSE_PARAM_NAMES = (
    'sigma_lo', 'sigma_hi', 'sigma_sat', 'e_lo', 'e_hi',
    'x0_lo', 'y0_lo', 'x0_hi', 'y0_hi', 'p_lo', 'p_hi', 'e1_lo', 'e2_lo', 'e1_hi', 'e2_hi',
    'au_lo', 'av_lo', 'au_hi', 'av_hi',
    'norm', 'e_pivot', 'alpha', 'beta',
)


def _log_energy_fraction(e, e_lo, e_hi, clip):
    """
    Position of e between e_lo (0) and e_hi (1) in log(e), optionally
    clipped to [0, 1]. 0 at all energies if e_lo == e_hi.
    """
    span = np.log(e_hi / e_lo)
    if span == 0:
        return np.zeros_like(e)

    t = np.log(e / e_lo) / span
    if clip:
        t = np.clip(t, 0, 1)

    return t


def energy_power_law(e, v_lo, v_hi, e_lo, e_hi, clip=False):
    """
    Positive quantity of the camera_response() spatial shape varying
    with energy as a power law through v_lo at e_lo and v_hi at e_hi,

        v(e) = v_lo * (v_hi/v_lo)**(log(e/e_lo) / log(e_hi/e_lo))

    Being monotonic, it stays between v_lo and v_hi for all energies
    between e_lo and e_hi, so bounding both bounds it over that whole
    range. If e_lo == e_hi, it is v_lo at all energies.

    Parameters
    ----------
    e: array_like
        Energy.
    v_lo, v_hi: array_like
        Values at e_lo and e_hi.
    e_lo, e_hi: array_like
        Reference energies, in the same unit as e.
    clip: bool
        If True, hold the value constant (v_lo or v_hi) outside of
        [e_lo, e_hi] instead of extrapolating the power law.

    Returns
    -------
    v: array_like
        Value at e.
    """
    return v_lo * np.power(v_hi / v_lo, _log_energy_fraction(e, e_lo, e_hi, clip))


def energy_saturating(e, v_lo, v_hi, e_lo, e_hi, rate):
    """
    Positive quantity of the camera_response() spatial shape varying
    with energy through v_lo at e_lo and v_hi at e_hi, with a change
    that saturates towards high energies:

        log v(e) = log v_lo + log(v_hi / v_lo) * g(t),
        g(t) = (1 - exp(-rate * t)) / (1 - exp(-rate))

    with t = log(e/e_lo) / log(e_hi/e_lo). rate = 0 (g = t) is the
    power law of energy_power_law(); for rate > 0 the change slows down
    with energy and v approaches v_lo * (v_hi/v_lo)**(1/(1 - exp(-rate)))
    at high energies, monotonically, so that its extrapolation above
    e_hi neither keeps growing as a power law nor turns over as a
    parabola in log-log space would. If v decreases (v_hi < v_lo), the
    saturation does not bound its extrapolation, so it is held at v_hi
    above e_hi instead. Below e_lo, v is held at v_lo. If e_lo == e_hi,
    it is v_lo at all energies.

    Parameters
    ----------
    e: array_like
        Energy.
    v_lo, v_hi: array_like
        Values at e_lo and e_hi.
    e_lo, e_hi: array_like
        Reference energies, in the same unit as e.
    rate: array_like
        Saturation rate, rate >= 0 (see above).

    Returns
    -------
    v: array_like
        Value at e.
    """
    t = np.maximum(_log_energy_fraction(e, e_lo, e_hi, clip=False), 0)
    if rate < 1e-8:
        g = t
    else:
        g = np.expm1(-rate * t) / np.expm1(-rate)
    if v_hi < v_lo:
        g = np.minimum(g, 1)

    return v_lo * np.power(v_hi / v_lo, g)


def energy_log_linear(e, v_lo, v_hi, e_lo, e_hi, clip=False):
    """
    Quantity of the camera_response() spatial shape varying linearly
    in log(e) from v_lo at e_lo to v_hi at e_hi. Unlike
    energy_power_law(), v_lo and v_hi may be 0 or negative. Being
    monotonic, it stays between v_lo and v_hi for all energies between
    e_lo and e_hi. If e_lo == e_hi, it is v_lo at all energies.

    Parameters
    ----------
    e: array_like
        Energy.
    v_lo, v_hi: array_like
        Values at e_lo and e_hi.
    e_lo, e_hi: array_like
        Reference energies, in the same unit as e.
    clip: bool
        If True, hold the value constant (v_lo or v_hi) outside of
        [e_lo, e_hi] instead of extrapolating.

    Returns
    -------
    v: array_like
        Value at e.
    """
    return v_lo + (v_hi - v_lo) * _log_energy_fraction(e, e_lo, e_hi, clip)


# Maximal eccentricity of the camera_response() spatial distribution,
# see ellipticity_to_ecc_phi().
MAX_ECCENTRICITY = 0.9

# Maximal asymmetry along the ellipse axes of the camera_response()
# spatial distribution (see super_gaussian()), in the fits. A larger
# asymmetry lets the fit shift the peak far to one side and pull the
# distribution back with a long tail on the other, mimicking a center
# offset (seen on real data at high energies, with peaks ~0.6 deg off
# the counts' centroid).
MAX_ASYMMETRY = 0.3


def ellipticity_to_ecc_phi(e1, e2):
    """
    Eccentricity and orientation (see super_gaussian()) from the
    Cartesian ellipticity components

        (e1, e2) = ecc * (cos(2*phi), sin(2*phi))

    used as camera_response() parameters. Unlike (ecc, phi), these are
    smooth through the circular case ecc = 0, where phi is undefined,
    so a fit does not get stuck there, and their interpolation in
    energy lets the orientation change with energy too. The
    eccentricity is capped at MAX_ECCENTRICITY.

    Parameters
    ----------
    e1, e2: array_like
        Ellipticity components.

    Returns
    -------
    ecc: array_like
        Eccentricity, 0 <= ecc <= MAX_ECCENTRICITY.
    phi: array_like
        Orientation in radians of the major axis with respect to the
        x axis, in (-pi/2, pi/2].
    """
    ecc = np.minimum(np.hypot(e1, e2), MAX_ECCENTRICITY)
    phi = 0.5 * np.arctan2(e2, e1)

    return ecc, phi


# Names of the parameters of a single-energy-bin fit of the spatial
# distribution (see CameraImage.fit_single_bins(), single_bin_response()).
SINGLE_BIN_PARAM_NAMES = ('x0', 'y0', 'sigma', 'p', 'e1', 'e2', 'a_u', 'a_v', 'norm')


def single_bin_response(x, y, x0, y0, sigma, p, e1, e2, a_u, a_v, norm):
    """
    Spatial background model of a single energy bin, fit on its own
    near the energy threshold (see CameraImage.fit_single_bins()): an
    elliptical, possibly asymmetric super-Gaussian (see
    super_gaussian(), ellipticity_to_ecc_phi()) times a normalization.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates, in degrees.
    x0, y0, sigma, p, e1, e2, a_u, a_v: array_like
        Center, width, exponent, ellipticity components and asymmetry,
        see camera_response().
    norm: array_like
        Normalization.

    Returns
    -------
    val: array_like
        Model value, in the units of camera_response().
    """
    ecc, phi = ellipticity_to_ecc_phi(e1, e2)
    return norm * super_gaussian(x, y, sigma, p, ecc=ecc, phi=phi, x0=x0, y0=y0, a_u=a_u, a_v=a_v)


def _minimize_with_restarts(fun, x0, bounds, max_restarts=5, tol=0.1):
    """L-BFGS-B minimization, restarted from its result until the
    objective improves by less than tol (see
    pybkgmodel.processing.restart_fit()). None if it does not converge."""
    options = dict(maxfun=50000, maxiter=50000)
    result = scipy.optimize.minimize(fun, x0, bounds=bounds, method='L-BFGS-B', options=options)
    if not result.success:
        return None
    for _ in range(max_restarts):
        restarted = scipy.optimize.minimize(fun, result.x, bounds=bounds, method='L-BFGS-B', options=options)
        if not restarted.success or restarted.fun > result.fun - tol:
            if restarted.success and restarted.fun < result.fun:
                result = restarted
            break
        result = restarted
    return result


def camera_response_params(result):
    """
    Fitted camera_response() parameters by name.

    Parameters
    ----------
    result: scipy.optimize.OptimizeResult
        Result of CameraImage.fit_camera_response().

    Returns
    -------
    params: dict
        Parameter values, keyed by CAMERA_RESPONSE_PARAM_NAMES. Widths
        and centers are in degrees, energies (e_lo, e_hi, e_pivot) in
        TeV.
    """
    return dict(zip(CAMERA_RESPONSE_PARAM_NAMES, result.x))


def camera_response(x, y, e,
                     sigma_lo, sigma_hi, sigma_sat, e_lo, e_hi,
                     x0_lo, y0_lo, x0_hi, y0_hi, p_lo, p_hi, e1_lo, e2_lo, e1_hi, e2_hi,
                     au_lo, av_lo, au_hi, av_hi,
                     norm, e_pivot, alpha, beta):
    """
    Camera background response model, defined as the product of:
    1. a spatial (x, y) distribution: a single elliptical 2D
       super-Gaussian (see super_gaussian()), normalized to integrate
       to 1 over the camera plane, whose width changes with energy,
    2. a linear function of y (currently fixed to a constant 1, see
       linear() -- both its slope and intercept are fixed for now, as
       this was found to make the fit more stable),
    3. a log-parabola function of e.

    The spatial distribution is centered at (x0, y0). Its width, its
    exponent p and its ellipticity vary with energy, each given by its
    values at e_lo and e_hi:
    - the width sigma with a change that saturates towards high
      energies at the rate sigma_sat (see energy_saturating()): a power
      law extrapolated above e_hi made the model too broad there,
      holding it constant above e_hi did not fit the energies below,
      and a parabola in log-log space turned over, narrowing the model
      at the highest energies,
    - p as a power law, held constant outside of [e_lo, e_hi],
    - the ellipticity components (e1, e2) linearly in log(e) (see
      energy_log_linear()), held constant outside of [e_lo, e_hi];
      they give the eccentricity and orientation, which can both
      change with energy (see ellipticity_to_ecc_phi()),
    - the center (x0, y0) and the asymmetry along the ellipse axes
      (a_u, a_v, see super_gaussian()) linearly in log(e), held
      constant outside of [e_lo, e_hi].
    p = 1 is a Gaussian; p < 1
    gives a more peaked shape with heavier wings, p > 1 a flat top with
    a steep edge.

    Parameters
    ----------
    x, y: array_like
        Camera plane coordinates.
    e: array_like
        Energy.
    sigma_lo, sigma_hi, sigma_sat, e_lo, e_hi: array_like
        Width of the spatial distribution at the energies e_lo and
        e_hi, and the saturation rate of its energy dependence, see
        energy_saturating().
    x0, y0: array_like
        Center of the spatial distribution.
    p_lo, p_hi: array_like
        Exponent of the spatial distribution (see super_gaussian()) at
        the energies e_lo and e_hi.
    e1_lo, e2_lo, e1_hi, e2_hi: array_like
        Ellipticity components of the spatial distribution at the
        energies e_lo and e_hi, see ellipticity_to_ecc_phi().
    au_lo, av_lo, au_hi, av_hi: array_like
        Asymmetry of the spatial distribution along its major (u) and
        minor (v) axes at the energies e_lo and e_hi, see
        super_gaussian().
    norm, e_pivot, alpha, beta: array_like
        Parameters of the log-parabola function of e (e_pivot being
        its reference energy e0), see log_parabola().

    Returns
    -------
    val: array_like
        Camera response value.
    """
    sigma = energy_saturating(e, sigma_lo, sigma_hi, e_lo, e_hi, sigma_sat)
    p = energy_power_law(e, p_lo, p_hi, e_lo, e_hi, clip=True)
    ecc, phi = ellipticity_to_ecc_phi(
        energy_log_linear(e, e1_lo, e1_hi, e_lo, e_hi, clip=True),
        energy_log_linear(e, e2_lo, e2_hi, e_lo, e_hi, clip=True),
    )
    x0 = energy_log_linear(e, x0_lo, x0_hi, e_lo, e_hi, clip=True)
    y0 = energy_log_linear(e, y0_lo, y0_hi, e_lo, e_hi, clip=True)
    a_u = energy_log_linear(e, au_lo, au_hi, e_lo, e_hi, clip=True)
    a_v = energy_log_linear(e, av_lo, av_hi, e_lo, e_hi, clip=True)
    spatial = super_gaussian(x, y, sigma, p=p, ecc=ecc, phi=phi, x0=x0, y0=y0, a_u=a_u, a_v=a_v)

    return (
        spatial
        * linear(y, a=0.0)
        * log_parabola(e, norm, e_pivot, alpha, beta)
    )


class EnergyPowerLaw:
    """
    Quantity varying with energy as the power law through the points
    (e1, v1) and (e2, v2), extrapolated beyond them, e.g. for a
    smoothing neighbourhood limit that grows with energy (see
    CameraImage.smoothed_posterior_rate()).

    Parameters
    ----------
    e1, e2: astropy.units.Quantity
        Energies of the two points.
    v1, v2: astropy.units.Quantity
        Values at e1 and e2.
    """

    def __init__(self, e1, v1, e2, v2):
        self.e1, self.v1, self.e2, self.v2 = e1, v1, e2, v2
        self.index = float(np.log((v2 / v1).decompose().value) / np.log((e2 / e1).decompose().value))

    def __call__(self, e):
        """Value at the energy (or energies) e."""
        return self.v1 * np.power((e / self.e1).decompose().value, self.index)

    def __repr__(self):
        return f"EnergyPowerLaw({self.v1} at {self.e1}, {self.v2} at {self.e2}; index {self.index:.3f})"


def at_energy(value, e):
    """
    Value of a quantity that may depend on energy: value(e) if it is
    callable (e.g. an EnergyPowerLaw), value itself otherwise.
    """
    return value(e) if callable(value) else value


def polar_neighbourhood(x, y, x0, y0, index, max_dphi, max_dr, max_dist=0*u.deg):
    """
    Pixels in the polar neighbourhood of one pixel, used for the
    moving average of polar_moving_average(). A pixel belongs to the
    neighbourhood of the pixel at index if either
    1. its center, seen from (x0, y0), lies within max_dphi in azimuth
       and within max_dr in radial distance of that pixel's center, or
       on the opposite side of (x0, y0) along the same line (azimuth
       within max_dphi of the opposite direction), where its radial
       distance counts as negative: a pixel at radial distance r_j
       across (x0, y0) from a pixel at r_i is r_i + r_j away radially.
       The neighbourhood is thus a strip along the line through
       (x0, y0), continuous across it, or
    2. its center lies within max_dist of that pixel's center,
       whatever its azimuth and radial distance (this keeps the
       neighbourhood from shrinking to a thin sliver near (x0, y0),
       where a fixed azimuth range spans very little).
    The pixel itself is always included.

    Parameters
    ----------
    x, y: astropy.units.Quantity
        Pixel center coordinates along x (nx values) and y (ny values).
    x0, y0: astropy.units.Quantity
        Center of the polar coordinate system.
    index: tuple of int
        (i, j) index of the pixel whose neighbourhood is returned.
    max_dphi: astropy.units.Quantity
        Maximal azimuth difference (angle between the lines joining
        (x0, y0) to the two pixel centers).
    max_dr: astropy.units.Quantity
        Maximal radial distance difference from (x0, y0).
    max_dist: astropy.units.Quantity
        Distance between pixel centers within which pixels are always
        included. Defaults to 0 (no such pixels).

    Returns
    -------
    neighbours: numpy.ndarray of bool
        Neighbourhood mask, of shape (nx, ny).
    """
    geometry = _polar_geometry(x, y, x0, y0)
    limits = _polar_limits(max_dphi, max_dr, max_dist)
    k = np.ravel_multi_index(index, (len(x), len(y)))

    return _polar_neighbours(geometry, k, limits).reshape((len(x), len(y)))


def _polar_geometry(x, y, x0, y0):
    """Flattened pixel center coordinates (deg) and polar coordinates around (x0, y0)."""
    xx, yy = np.meshgrid(x.to_value(u.deg), y.to_value(u.deg), indexing='ij')
    dx = xx.ravel() - x0.to_value(u.deg)
    dy = yy.ravel() - y0.to_value(u.deg)

    return xx.ravel(), yy.ravel(), np.hypot(dx, dy), np.arctan2(dy, dx)


def _polar_limits(max_dphi, max_dr, max_dist):
    """Neighbourhood limits in rad / deg / deg."""
    # Small tolerance, so that pixels exactly on the neighbourhood
    # boundary (common on a regular grid) are not lost to round-off.
    tol = 1e-9

    return max_dphi.to_value(u.rad) + tol, max_dr.to_value(u.deg) + tol, max_dist.to_value(u.deg) + tol


def _polar_neighbours(geometry, k, limits):
    """Flattened neighbourhood mask of the k-th (flattened) pixel."""
    xx, yy, r, phi = geometry
    dphi_lim, dr_lim, dist_lim = limits

    dphi = (phi - phi[k] + np.pi) % (2*np.pi) - np.pi
    same_side = (np.abs(dphi) <= dphi_lim) & (np.abs(r - r[k]) <= dr_lim)
    # Across (x0, y0), the radial distance counts as negative (-r).
    opposite_side = (np.abs(dphi) >= np.pi - dphi_lim) & (r + r[k] <= dr_lim)
    polar = same_side | opposite_side
    close = np.hypot(xx - xx[k], yy - yy[k]) <= dist_lim

    return polar | close


def polar_moving_average(values, x, y, x0, y0, max_dphi, max_dr, max_dist=0*u.deg, weights=None):
    """
    Moving average of a camera map over polar neighbourhoods around
    the point (x0, y0): for each pixel, the new value is the mean over
    the pixels in its neighbourhood (see polar_neighbourhood()),
    optionally weighted.

    Parameters
    ----------
    values: array_like or astropy.units.Quantity
        Map(s) to average, of shape (..., nx, ny).
    x, y: astropy.units.Quantity
        Pixel center coordinates along x (nx values) and y (ny values).
    x0, y0: astropy.units.Quantity
        Center of the polar coordinate system.
    max_dphi, max_dr, max_dist: astropy.units.Quantity
        Neighbourhood limits, see polar_neighbourhood().
    weights: array_like, optional
        Weight of each pixel, of shape (nx, ny), e.g. 0 for pixels to
        leave out of the averages. Defaults to equal weights.

    Returns
    -------
    averaged: array_like or astropy.units.Quantity
        Averaged map(s), of the same shape and unit as values. NaN for
        pixels whose neighbourhood has zero total weight.
    """
    unit = getattr(values, 'unit', None)
    values = np.asarray(getattr(values, 'value', values), dtype=float)

    geometry = _polar_geometry(x, y, x0, y0)
    limits = _polar_limits(max_dphi, max_dr, max_dist)

    flat = values.reshape(values.shape[:-2] + (-1,))
    averaged = np.empty_like(flat)
    w = np.ones(flat.shape[-1]) if weights is None else np.asarray(weights, dtype=float).ravel()
    # Pixels left out (zero weight) may hold non-finite values.
    flat = np.where(w > 0, flat, 0.0)

    for k in range(flat.shape[-1]):
        neighbours = _polar_neighbours(geometry, k, limits)
        total = w[neighbours].sum()
        if total > 0:
            averaged[..., k] = (flat[..., neighbours] * w[neighbours]).sum(axis=-1) / total
        else:
            averaged[..., k] = np.nan

    averaged = averaged.reshape(values.shape)
    if unit is not None:
        averaged = averaged * unit

    return averaged


def merged_bin_starts(edges, min_width):
    """
    Start indices of groups of consecutive bins to merge, so that every
    merged bin is at least min_width wide (to be used with
    numpy.add.reduceat). Bins are merged only if they are narrower than
    min_width; a leftover group of fewer bins at the end is merged into
    the preceding group. Assumes uniform bins.

    Parameters
    ----------
    edges: astropy.units.Quantity
        Uniform bin edges.
    min_width: astropy.units.Quantity
        Minimal width of the merged bins.

    Returns
    -------
    starts: numpy.ndarray
        Index of the first bin of each group.
    """
    n = len(edges) - 1
    # Small tolerance, so that e.g. 3 bins of 0.1 deg count as 0.3 deg.
    ratio = (min_width / (edges[1] - edges[0])).decompose().value
    k = max(1, int(np.ceil(ratio - 1e-9)))

    starts = np.arange(0, n, k)
    if len(starts) > 1 and n - starts[-1] < k:
        starts = starts[:-1]

    return starts


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
            exposure = np.ones((nx, ny), dtype=float) * u.s
        elif exposure.shape == ():
            exposure = np.repeat(exposure, nx * ny).reshape((nx, ny))

        self.raw_counts = counts
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center
        self.mask = mask
        self.raw_exposure = exposure
        # Radius of the field of view, see mask_outside_fov(); None
        # means the whole binned area.
        self.fov_radius = None

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

        int2diff = self._int2diff(index)

        # Using model_rate (exposure-independent) directly, rather than
        # model_counts / exposure, avoids a 0/0 division in pixels with
        # zero exposure (e.g. never covered by any stacked run/always
        # excluded); model_rate is mathematically identical to
        # model_counts / exposure wherever exposure is nonzero.
        rate = result.model_rate / self.pixel_area
        dnde = rate * int2diff[:, None, None]

        return dnde, result

    def low_count_bins(self, peak_fraction):
        """
        Energy bins below the bin with the highest total count (in the
        unmasked pixels), holding less than peak_fraction of its counts:
        near and below the energy threshold, where the spatial
        distribution departs from the fitted camera_response() shape
        (see smoothed_posterior_rate()).

        Parameters
        ----------
        peak_fraction: float
            Fraction of the peak bin's counts.

        Returns
        -------
        low: numpy.ndarray of bool
            Whether each energy bin is such a low-count bin.
        """
        mask = np.broadcast_to(self.mask, self.raw_counts.shape)
        bin_totals = (self.raw_counts * mask).sum(axis=(1, 2))
        peak_bin = np.argmax(bin_totals)

        return (np.arange(len(bin_totals)) < peak_bin) & (bin_totals < peak_fraction * bin_totals[peak_bin])

    def single_fit_bins(self, peak_fraction):
        """
        Energy bins below the bin with the highest total count (in the
        unmasked pixels), holding at least peak_fraction of its counts:
        just above the energy threshold, where the spatial distribution
        departs from the shape of the fitted camera_response() but the
        counts suffice to fit each bin on its own (see
        fit_single_bins()).

        Parameters
        ----------
        peak_fraction: float
            Fraction of the peak bin's counts.

        Returns
        -------
        selected: numpy.ndarray of bool
            Whether each energy bin is such a bin.
        """
        mask = np.broadcast_to(self.mask, self.raw_counts.shape)
        bin_totals = (self.raw_counts * mask).sum(axis=(1, 2))
        peak_bin = np.argmax(bin_totals)

        return ((np.arange(len(bin_totals)) < peak_bin)
                & (bin_totals >= peak_fraction * bin_totals[peak_bin]) & (bin_totals > 0))

    def data_only_bins(self, result, peak_fraction):
        """
        Energy bins whose background model is the observed rate smoothed
        alone (see smoothed_posterior_rate()): the low-count bins (see
        low_count_bins()), and the bins to be fit on their own (see
        single_fit_bins()) whose fit did not converge.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), after fit_single_bins() if
            it was run.
        peak_fraction: float
            Fraction of the peak bin's counts, see low_count_bins().

        Returns
        -------
        selected: numpy.ndarray of bool
            Whether each energy bin is such a bin.
        """
        selected = self.low_count_bins(peak_fraction)
        fitted = getattr(result, 'single_bin_fits', {})
        for i in np.nonzero(self.single_fit_bins(peak_fraction))[0]:
            if int(i) not in fitted:
                selected[i] = True

        return selected

    def fit_energy_range(self, min_counts=1000):
        """
        Energy bins included in fit_camera_response(): the bin with the
        highest total count (in the unmasked pixels) and all bins above
        it. Bins below the peak carry little information to constrain
        the energy-dependent fit parameters and would otherwise just
        add spurious -model_counts terms to the fit statistic.

        Parameters
        ----------
        min_counts: float
            Minimal total count (in the unmasked pixels) of an energy
            bin for it to count as constraining the spatial shape,
            see e_lo, e_hi below. Defaults to 1000, so that the shape
            parameters given at e_lo and e_hi (see camera_response())
            are set where the data constrain them well rather than
            extrapolated from a few events.

        Returns
        -------
        in_range: numpy.ndarray of bool
            Whether each energy bin is included in the fit.
        e_lo, e_hi: float
            Centers (in TeV) of the lowest and highest included energy
            bins holding at least min_counts counts, i.e. the energy
            range where the data actually constrain the spatial shape
            (a few stray events at the highest energies do not extend
            it). If no included bin has that many counts, both are the
            peak bin's center.
        """
        mask = np.broadcast_to(self.mask, self.raw_counts.shape)
        bin_totals = (self.raw_counts * mask).sum(axis=(1, 2))
        peak_bin = np.argmax(bin_totals)
        in_range = np.arange(len(bin_totals)) >= peak_bin

        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(u.TeV)
        filled = np.nonzero(in_range & (bin_totals >= min_counts))[0]
        if filled.size == 0:
            filled = np.array([peak_bin])

        return in_range, e[filled[0]], e[filled[-1]]

    def fit_camera_response(self, p0, bounds=None, method='L-BFGS-B', **kwargs):
        """
        Fit the product of camera_response() and the exposure map to the
        observed counts, assuming the counts follow Poisson statistics
        (i.e. minimizing the C-stat, see cstat()).

        Only the unmasked pixels (see mask, mask_half(), mask_region())
        are included in the fit. In energy, only the bins selected by
        fit_energy_range() are included.

        Parameters
        ----------
        p0: array_like
            Initial guess for the camera_response() fit parameters
            (sigma_lo, sigma_hi, sigma_sat, e_lo, e_hi, x0_lo, y0_lo, x0_hi, y0_hi, p_lo, p_hi,
            e1_lo, e2_lo, e1_hi, e2_hi, au_lo, av_lo, au_hi, av_hi,
            norm, e_pivot, alpha, beta), see CAMERA_RESPONSE_PARAM_NAMES.
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
            map of the same shape as CameraImage.counts -- additionally
            rescaled per energy bin so that its sum over the unmasked
            pixels exactly equals the observed counts' sum over the
            same pixels (see mask, mask_half(), mask_region()), and set
            to 0 outside of the field of view (see mask_outside_fov()).
        """
        x = ((self.xedges[1:] + self.xedges[:-1]) / 2).to_value(u.deg)
        y = ((self.yedges[1:] + self.yedges[:-1]) / 2).to_value(u.deg)
        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(u.TeV)

        ee, xx, yy = np.meshgrid(e, x, y, indexing='ij')

        exposure = self.raw_exposure.to_value(u.s)
        counts = self.raw_counts
        mask = np.broadcast_to(self.mask, counts.shape)

        in_range, _, _ = self.fit_energy_range()
        mask = mask & in_range[:, None, None]

        def neg_log_likelihood(params):
            model_counts = camera_response(xx, yy, ee, *params) * exposure
            # Floor the model, which can underflow to 0 far from the
            # camera center (e.g. for a steep super-Gaussian tail),
            # where log(0) would make the fit statistic NaN.
            model_counts = np.maximum(model_counts, 1e-30)
            return cstat(counts[mask], model_counts[mask])

        result = scipy.optimize.minimize(
            neg_log_likelihood,
            x0=p0,
            bounds=bounds,
            method=method,
            **kwargs
        )

        # The model is 0 outside of the field of view (see
        # mask_outside_fov()).
        inside_fov = self.inside_fov[None, :, :]
        result.model_counts = camera_response(xx, yy, ee, *result.x) * exposure * inside_fov

        # Rescale the model, per energy bin, so that its sum over the
        # unmasked pixels exactly matches the observed counts' sum
        # over the same pixels. This removes any residual mismatch
        # left by the smooth log_parabola energy shape not perfectly
        # tracking the true spectrum bin-by-bin, while leaving the
        # fitted spatial shape untouched. Uses the full pixel mask
        # (not the fit's additional peak-and-above energy restriction
        # above), so every energy bin is normalized this way.
        mask_full = np.broadcast_to(self.mask, counts.shape)
        data_sum = (counts * mask_full).sum(axis=(1, 2))
        model_sum = (result.model_counts * mask_full).sum(axis=(1, 2))
        scale = np.where(model_sum > 0, data_sum / np.where(model_sum > 0, model_sum, 1), 1.0)
        result.model_counts = result.model_counts * scale[:, None, None]

        # Exposure-independent fitted rate (model_counts = model_rate *
        # exposure): unlike model_counts, this stays well-defined even
        # in pixels with zero exposure, and is what fitted_differential_rate()
        # / posterior_differential_rate() fall back to there instead of
        # dividing by zero. The same per-energy-bin rescaling as
        # model_counts above is applied here too, for consistency.
        result.model_rate = camera_response(xx, yy, ee, *result.x) * inside_fov / u.s * scale[:, None, None]

        return result

    def fit_single_bins(self, result, peak_fraction):
        """
        Fit the energy bins just above the energy threshold (see
        single_fit_bins()) each on its own, with an elliptical,
        possibly asymmetric super-Gaussian (see single_bin_response()),
        and replace the model of fit_camera_response() in these bins
        with it. The global model, which does not include these bins in
        its fit, does not follow their spatial distribution (e.g. a
        center shifted by ~0.7 deg at large zenith angles).

        Each bin is fit from the first and second moments of its counts,
        directly and in two stages (symmetric first), keeping the better
        fit, as for the global model (see
        pybkgmodel.processing.fit_camera_response_two_ways()). As in
        fit_camera_response(), the model is 0 outside of the field of
        view and rescaled to the observed counts in the unmasked pixels.
        Bins whose fit does not converge keep the global model, and are
        smoothed from the data alone (see data_only_bins()).

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), updated in place:
            model_counts and model_rate of the fitted bins are replaced,
            and result.single_bin_fits maps each fitted bin's index to
            its parameters (a dict keyed by SINGLE_BIN_PARAM_NAMES).
        peak_fraction: float
            Fraction of the peak bin's counts, see single_fit_bins().

        Returns
        -------
        result: scipy.optimize.OptimizeResult
            The updated result.
        """
        result.single_bin_fits = {}

        x = ((self.xedges[1:] + self.xedges[:-1]) / 2).to_value(u.deg)
        y = ((self.yedges[1:] + self.yedges[:-1]) / 2).to_value(u.deg)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        exposure = self.raw_exposure.to_value(u.s)
        inside_fov = self.inside_fov
        mask = self.mask
        pixel_area = (np.diff(self.xedges)[0] * np.diff(self.yedges)[0]).to_value(u.deg**2)

        x_min, x_max = self.xedges[0].to_value(u.deg), self.xedges[-1].to_value(u.deg)
        y_min, y_max = self.yedges[0].to_value(u.deg), self.yedges[-1].to_value(u.deg)
        x_range = x_max - x_min
        i_asym = [SINGLE_BIN_PARAM_NAMES.index(name) for name in ('a_u', 'a_v')]

        for i in np.nonzero(self.single_fit_bins(peak_fraction))[0]:
            counts = self.raw_counts[i] * mask
            total = counts.sum()

            # Starting values from the counts' moments.
            cx, cy = (counts * xx).sum() / total, (counts * yy).sum() / total
            cxx = (counts * (xx - cx)**2).sum() / total
            cyy = (counts * (yy - cy)**2).sum() / total
            cxy = (counts * (xx - cx) * (yy - cy)).sum() / total
            eigenvalues = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
            sigma0 = np.clip(np.sqrt(np.sqrt(max(eigenvalues[0], 1e-6) * eigenvalues[1])), x_range / 100, x_range)
            ecc0 = np.clip(1 - np.sqrt(max(eigenvalues[0], 0) / eigenvalues[1]), 0.02, 0.8)
            phi0 = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
            norm0 = total * pixel_area / exposure[mask].mean()

            p0 = [cx, cy, sigma0, 1.0, ecc0 * np.cos(2 * phi0), ecc0 * np.sin(2 * phi0), 0.0, 0.0, norm0]
            bounds = [
                (x_min, x_max), (y_min, y_max), (x_range / 100, x_range), (0.3, 5.0),
                (-MAX_ECCENTRICITY, MAX_ECCENTRICITY), (-MAX_ECCENTRICITY, MAX_ECCENTRICITY),
                (-MAX_ASYMMETRY, MAX_ASYMMETRY), (-MAX_ASYMMETRY, MAX_ASYMMETRY), (norm0 / 1e3, norm0 * 1e3),
            ]
            symmetric_bounds = [(0.0, 0.0) if k in i_asym else b for k, b in enumerate(bounds)]

            def neg_log_likelihood(q, counts=counts):
                model = np.maximum(single_bin_response(xx, yy, *q) * exposure, 1e-30)
                return cstat(counts[mask], model[mask])

            candidates = [_minimize_with_restarts(neg_log_likelihood, p0, bounds)]
            symmetric = _minimize_with_restarts(neg_log_likelihood, p0, symmetric_bounds)
            if symmetric is not None:
                candidates.append(_minimize_with_restarts(neg_log_likelihood, symmetric.x, bounds))
            candidates = [c for c in candidates if c is not None and np.isfinite(c.fun)]
            if not candidates:
                continue
            best = min(candidates, key=lambda c: c.fun)

            rate = single_bin_response(xx, yy, *best.x) * inside_fov
            model_counts = rate * exposure
            model_sum = (model_counts * mask).sum()
            scale = total / model_sum if model_sum > 0 else 1.0

            result.model_counts[i] = model_counts * scale
            result.model_rate[i] = rate * scale / u.s
            result.single_bin_fits[int(i)] = dict(zip(SINGLE_BIN_PARAM_NAMES, (float(v) for v in best.x)))

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
        prior_strength: float or array_like
            Equivalent number of prior "pseudo-counts" k (k > 0)
            controlling how strongly the fitted model constrains the
            posterior relative to the data. Larger values weight the
            prior (the fit) more heavily; smaller values let the raw
            data dominate. Either a single value for all bins, or one
            per camera pixel (shape (nx, ny), the same for all energy
            bins), e.g. proportional to the pixel solid angle so that
            the relative weight does not depend on the binning.
            Defaults to 1.0.

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
        prior_strength: float or array_like
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
        prior_strength: float or array_like
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
        int2diff = self._int2diff(index)

        rate = self.posterior_rate(result, prior_strength=prior_strength) / self.pixel_area
        dnde = rate * int2diff[:, None, None]

        return dnde

    def _int2diff(self, index):
        """
        Conversion factor, per energy bin, from counts integrated over
        the bin to the differential value at e0 = (emin * emax)**0.5,
        assuming a power law of the given spectral index (see
        differential_rate()).
        """
        emin = self.energy_edges[:-1]
        emax = self.energy_edges[1:]
        e0 = (emin * emax)**0.5
        return (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))

    def smoothed_posterior_rate(self, result, prior_strength=1.0, max_dphi=10*u.deg, max_dr=0.25*u.deg, max_dist=0.25*u.deg, data_peak_fraction=None, energy_bins=None):
        """
        Bayesian posterior rate (see posterior_rate()), smoothed by a
        moving average over polar neighbourhoods around the fitted
        center (x0, y0) of camera_response(),
        see polar_moving_average().

        The plain average would flatten the steep radial profile of
        the background (lowering the center, raising the outskirts).
        To avoid this, the averaged posterior of each pixel i is
        corrected with the fitted model rate f (result.model_rate):

            smoothed_i = <posterior>_i * f_i / <f>_i

        where <.>_i is the average over the neighbourhood of pixel i.
        A posterior following the fitted shape is thus left unchanged,
        and only deviations from that shape are smoothed. Finally, the
        smoothed rate is rescaled, per energy bin, so that its sum over
        all pixels equals that of the posterior rate.

        All averages are taken over the rate per unit solid angle, so
        that pixels of slightly different solid angle are weighted
        consistently.

        If data_peak_fraction is given, the low-count energy bins below
        the count peak (see low_count_bins()), near and below the
        energy threshold, are treated differently: there the spatial
        distribution departs from the fitted shape, so neither the
        Bayesian update nor the shape correction is used. Their
        observed rate is smoothed by a plain moving average over the
        unmasked pixels, masked pixels being filled from their unmasked
        neighbours (see _data_smoothed_rate()).

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), see posterior_counts().
        prior_strength: float or array_like
            Prior weight, see posterior_counts().
        max_dphi: astropy.units.Quantity or EnergyPowerLaw
            Maximal azimuth difference, seen from the model center, of
            the pixels averaged together, or its dependence on energy,
            evaluated at each energy bin's center energy (see
            at_energy()). Defaults to 10 deg.
        max_dr: astropy.units.Quantity or EnergyPowerLaw
            Maximal difference in radial distance from the model center
            of the pixels averaged together, counted as negative across
            the peak (see polar_neighbourhood()). Defaults to 0.25 deg.
        max_dist: astropy.units.Quantity or EnergyPowerLaw
            Distance between pixel centers within which pixels are
            always averaged together, whatever max_dphi and max_dr
            (see polar_neighbourhood()). Defaults to 0.25 deg.
        data_peak_fraction: float, optional
            If given, the energy bins below the count peak holding less
            than this fraction of its counts are smoothed from the data
            alone (see above). Defaults to None (not used).
        energy_bins: array_like of int, optional
            Indices of the energy bins to compute, e.g. to plot a single
            one. Defaults to None (all energy bins).

        Returns
        -------
        rate: array_like astropy.unit.Quantity
            Smoothed posterior rate, of the same shape as
            CameraImage.counts (or of the selected energy bins only, see
            energy_bins).
        """
        n_e = len(self.energy_edges) - 1
        selected = np.arange(n_e) if energy_bins is None else np.atleast_1d(energy_bins)

        posterior = self.posterior_rate(result, prior_strength=prior_strength)[selected]
        posterior_density = posterior / self.pixel_area
        model_density = result.model_rate[selected] / self.pixel_area

        averaged = self._centered_moving_average(result, posterior_density, selected, max_dphi, max_dr, max_dist)
        model_averaged = self._centered_moving_average(result, model_density, selected, max_dphi, max_dr, max_dist)

        # Shape correction f_i / <f>_i; left out (i.e. 1) where the
        # neighbourhood model average vanishes.
        valid = model_averaged.value > 0
        with np.errstate(invalid='ignore', divide='ignore'):
            correction = np.where(
                valid,
                (model_density / np.where(valid, model_averaged, 1 * model_averaged.unit)).decompose().value,
                1.0
            )

        smoothed = averaged * correction * self.pixel_area

        # Per-energy-bin renormalization to the posterior total.
        posterior_sum = posterior.sum(axis=(1, 2))
        smoothed_sum = smoothed.sum(axis=(1, 2))
        valid = smoothed_sum.value > 0
        scale = np.where(
            valid,
            (posterior_sum / np.where(valid, smoothed_sum, 1 * smoothed_sum.unit)).decompose().value,
            1.0
        )
        smoothed = smoothed * scale[:, None, None]

        if data_peak_fraction is not None:
            low = self.data_only_bins(result, data_peak_fraction)[selected]
            if low.any():
                smoothed[low] = self._data_smoothed_rate(
                    result, selected[low], max_dphi=max_dphi, max_dr=max_dr, max_dist=max_dist
                )

        return smoothed

    def _centered_moving_average(self, result, values, energy_bins, max_dphi, max_dr, max_dist, weights=None):
        """
        polar_moving_average() of each of the given energy bins' maps
        (values, of shape (len(energy_bins), nx, ny)), around the model
        center of that bin (see _bin_centers()), with the neighbourhood
        limits at that bin's center energy (see at_energy()).
        """
        x = (self.xedges[1:] + self.xedges[:-1]) / 2
        y = (self.yedges[1:] + self.yedges[:-1]) / 2
        x0, y0 = self._bin_centers(result, energy_bins)
        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1])[energy_bins]

        return u.Quantity([
            polar_moving_average(
                values[i], x, y, x0[i], y0[i],
                at_energy(max_dphi, e[i]), at_energy(max_dr, e[i]), at_energy(max_dist, e[i]),
                weights=weights
            )
            for i in range(len(energy_bins))
        ])

    def _data_smoothed_rate(self, result, energy_bins, max_dphi, max_dr, max_dist):
        """
        Observed rate of the given energy bins, smoothed by a plain
        moving average over the unmasked pixels only (see
        smoothed_posterior_rate()). Masked pixels are filled with the
        average of their unmasked neighbours, or with the fitted model
        rate where there is none; pixels outside of the field of view
        (see mask_outside_fov()) are 0. The result is renormalized, per
        energy bin, so that its sum over the unmasked pixels equals
        that of the observed rate.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            data_rate = self.counts[energy_bins] / self.raw_exposure
        data_rate = np.where(self.mask, data_rate, 0 * data_rate.unit)
        model_density = result.model_rate[energy_bins] / self.pixel_area

        averaged = self._centered_moving_average(
            result, data_rate / self.pixel_area, energy_bins, max_dphi, max_dr, max_dist,
            weights=self.mask
        )
        averaged = np.where(np.isfinite(averaged.value), averaged, model_density)
        smoothed = averaged * self.pixel_area * self.inside_fov

        data_sum = data_rate.sum(axis=(1, 2))
        smoothed_sum = (smoothed * self.mask).sum(axis=(1, 2))
        valid = smoothed_sum.value > 0
        scale = np.where(
            valid,
            (data_sum / np.where(valid, smoothed_sum, 1 * smoothed_sum.unit)).decompose().value,
            1.0
        )

        return smoothed * scale[:, None, None]

    def smoothed_posterior_differential_rate(self, result, prior_strength=1.0, max_dphi=10*u.deg, max_dr=0.25*u.deg, max_dist=0.25*u.deg, data_peak_fraction=None, index=-2):
        """
        Same as posterior_differential_rate(), but based on the
        smoothed posterior rate of smoothed_posterior_rate().

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), see posterior_counts().
        prior_strength: float or array_like
            Prior weight, see posterior_counts().
        max_dphi, max_dr, max_dist: astropy.units.Quantity
            Smoothing neighbourhood, see smoothed_posterior_rate().
        data_peak_fraction: float, optional
            Low-count energy bins smoothed from the data alone, see
            smoothed_posterior_rate().
        index: float
            Power law spectral index, see differential_rate().
            Defaults to -2.

        Returns
        -------
        dnde: array_like astropy.unit.Quantity
            Smoothed posterior differential rate, of the same shape as
            the camera image.
        """
        rate = self.smoothed_posterior_rate(
            result, prior_strength=prior_strength, max_dphi=max_dphi, max_dr=max_dr, max_dist=max_dist,
            data_peak_fraction=data_peak_fraction
        )

        return rate / self.pixel_area * self._int2diff(index)[:, None, None]

    def _bin_centers(self, result, energy_bins):
        """Model center (x0, y0) of each of the given energy bins: that
        of its own fit for the bins fit on their own (see
        fit_single_bins()), that of the global model at the bin's
        energy otherwise (see _model_center())."""
        e = np.sqrt(self.energy_edges[1:] * self.energy_edges[:-1]).to_value(u.TeV)[energy_bins]
        x0, y0 = self._model_center(result, e)
        x0, y0 = np.array(x0.to_value(u.deg), ndmin=1), np.array(y0.to_value(u.deg), ndmin=1)

        fits = getattr(result, 'single_bin_fits', {})
        for k, i in enumerate(energy_bins):
            if int(i) in fits:
                x0[k], y0[k] = fits[int(i)]['x0'], fits[int(i)]['y0']

        return x0 * u.deg, y0 * u.deg

    @staticmethod
    def _model_center(result, e):
        """Fitted center (x0, y0) of the camera_response() spatial
        distribution at the energies e (in TeV), see camera_response()."""
        params = camera_response_params(result)
        e_range = (params['e_lo'], params['e_hi'])
        x0 = energy_log_linear(e, params['x0_lo'], params['x0_hi'], *e_range, clip=True)
        y0 = energy_log_linear(e, params['y0_lo'], params['y0_hi'], *e_range, clip=True)

        return x0 * u.deg, y0 * u.deg

    def plot_smoothing_neighbourhood(self, result, max_dphi=10*u.deg, max_dr=0.25*u.deg, max_dist=0.25*u.deg, pixels=None, energy_bin_id=None, ax_unit='deg'):
        """
        Illustrate the smoothing neighbourhood of smoothed_posterior_rate()
        (see polar_neighbourhood()) for example pixels: the pixels
        averaged together are filled, the example pixel is outlined,
        and the polar (max_dphi, max_dr) and distance (max_dist)
        limits are drawn as dashed lines around it. Energy-dependent
        limits (see at_energy()) are taken at the energy of the counts
        shown, or at e_lo.

        The fitted camera_response() spatial distribution is overlaid
        as its sigma ellipse (semi-axes sigma / sqrt(1 - ecc) and
        sigma * sqrt(1 - ecc), rotated by phi, see super_gaussian()),
        at both ends e_lo and e_hi of the fitted energy range.

        If energy_bin_id is given, the observed counts of that energy
        bin are shown underneath (masked pixels blank), the averaged
        pixels are outlined instead of filled so the counts stay
        visible, and the sigma ellipse is drawn at the energy bin's
        center only (see camera_response()). Each ellipse is centered at
        the model center at its energy.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), giving the model center.
        max_dphi, max_dr, max_dist: astropy.units.Quantity
            Neighbourhood limits, see smoothed_posterior_rate().
        pixels: list of tuple of int, optional
            (i, j) indices of the example pixels. Defaults to two
            pixels: the one closest to the model center, and one far from
            it (at 70% of the smaller half-extent of the camera, toward
            the upper right).
        energy_bin_id: int, optional
            Energy bin whose counts are shown underneath. If None
            (default), no counts are shown and the sigma ellipse is
            drawn at e_lo and e_hi.
        ax_unit: str
            Unit to use for the x/y axes.

        Returns
        -------
        fig, ax: matplotlib figure and axes
        """
        from matplotlib import patheffects
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Circle, Ellipse, Rectangle, Wedge

        # The neighbourhoods are drawn around the model center at the
        # energy of the counts shown, or at e_lo without counts, with
        # the neighbourhood limits at that energy.
        if energy_bin_id is not None:
            x0, y0 = self._bin_centers(result, [energy_bin_id])
            x0, y0 = x0[0], y0[0]
            e_limits = np.sqrt(self.energy_edges[energy_bin_id] * self.energy_edges[energy_bin_id + 1])
        else:
            x0, y0 = self._model_center(result, camera_response_params(result)['e_lo'])
            e_limits = camera_response_params(result)['e_lo'] * u.TeV
        max_dphi = at_energy(max_dphi, e_limits)
        max_dr = at_energy(max_dr, e_limits)
        max_dist = at_energy(max_dist, e_limits)
        x = (self.xedges[1:] + self.xedges[:-1]) / 2
        y = (self.yedges[1:] + self.yedges[:-1]) / 2
        xx, yy = np.meshgrid(x.to_value(u.deg), y.to_value(u.deg), indexing='ij')

        if pixels is None:
            half = min(np.ptp(self.xedges), np.ptp(self.yedges)).to_value(u.deg) / 2
            far = (x0.to_value(u.deg) + 0.7*half*np.cos(np.pi/4), y0.to_value(u.deg) + 0.7*half*np.sin(np.pi/4))
            pixels = [
                np.unravel_index(np.argmin(np.hypot(xx - x0.to_value(u.deg), yy - y0.to_value(u.deg))), xx.shape),
                np.unravel_index(np.argmin(np.hypot(xx - far[0], yy - far[1])), xx.shape),
            ]

        # Categorical slots of the reference palette: 1 and 2 (blue,
        # orange) for the example pixels; on top of a counts map, 3
        # and 5 (aqua, magenta) for the fitted components, otherwise
        # black.
        colors = ['#2a78d6', '#eb6834']
        on_counts = energy_bin_id is not None
        shape_color = '#1baf7a' if on_counts else 'black'

        to_unit = lambda q: q.to_value(ax_unit)
        xedges = to_unit(self.xedges)
        yedges = to_unit(self.yedges)
        fig, ax = pyplot.subplots(figsize=(7.5 if on_counts else 7, 7 * np.ptp(self.yedges) / np.ptp(self.xedges) + 0.5))

        if on_counts:
            counts = np.where(self.mask, self.counts[energy_bin_id], np.nan)
            im = ax.pcolormesh(xedges, yedges, counts.T, cmap='Greys', vmin=0)
            fig.colorbar(im, ax=ax, label='Counts', fraction=0.046, pad=0.04)
        else:
            # Faint pixel grid.
            ax.pcolormesh(
                xedges, yedges, np.zeros((len(y), len(x))),
                cmap=ListedColormap(['white']), edgecolors='#e0e0e0', linewidth=0.5
            )

        for index, color in zip(pixels, colors * len(pixels)):
            index = tuple(int(v) for v in index)
            neighbours = polar_neighbourhood(x, y, x0, y0, index, max_dphi, max_dr, max_dist)

            if on_counts:
                # Outline of the averaged pixels: pixel sides not shared
                # with another averaged pixel.
                padded = np.pad(neighbours, 1)
                segments = []
                for i, j in zip(*np.nonzero(neighbours)):
                    if not padded[i, j+1]:
                        segments.append([(xedges[i], yedges[j]), (xedges[i], yedges[j+1])])
                    if not padded[i+2, j+1]:
                        segments.append([(xedges[i+1], yedges[j]), (xedges[i+1], yedges[j+1])])
                    if not padded[i+1, j]:
                        segments.append([(xedges[i], yedges[j]), (xedges[i+1], yedges[j])])
                    if not padded[i+1, j+2]:
                        segments.append([(xedges[i], yedges[j+1]), (xedges[i+1], yedges[j+1])])
                ax.add_collection(LineCollection(segments, colors=color, linewidths=2))
            else:
                ax.pcolormesh(
                    xedges, yedges,
                    np.ma.masked_where(~neighbours.T, np.ones(neighbours.T.shape)),
                    cmap=ListedColormap([color]), alpha=0.45
                )

            i, j = index
            ax.add_patch(Rectangle(
                (xedges[i], yedges[j]), xedges[i+1] - xedges[i], yedges[j+1] - yedges[j],
                fill=False, edgecolor=color, linewidth=2, hatch='////' if on_counts else None
            ))

            # Limits: annular sector (polar condition) and circle (distance condition).
            dx = (x[i] - x0).to_value(u.deg)
            dy = (y[j] - y0).to_value(u.deg)
            r = np.hypot(dx, dy)
            phi = np.degrees(np.arctan2(dy, dx))
            r_out = r + max_dr.to_value(u.deg)
            r_in = max(0.0, r - max_dr.to_value(u.deg))
            dphi = max_dphi.to_value(u.deg)
            ax.add_patch(Wedge(
                (to_unit(x0), to_unit(y0)), (r_out * u.deg).to_value(ax_unit), phi - dphi, phi + dphi,
                width=((r_out - r_in) * u.deg).to_value(ax_unit),
                fill=False, edgecolor=color, linestyle='--', linewidth=1.2
            ))
            # Part of the strip across the peak (negative radial distance).
            r_across = max_dr.to_value(u.deg) - r
            if r_across > 0:
                ax.add_patch(Wedge(
                    (to_unit(x0), to_unit(y0)), (r_across * u.deg).to_value(ax_unit),
                    phi + 180 - dphi, phi + 180 + dphi,
                    fill=False, edgecolor=color, linestyle='--', linewidth=1.2
                ))
            ax.add_patch(Circle(
                (to_unit(x[i]), to_unit(y[j])), to_unit(max_dist),
                fill=False, edgecolor=color, linestyle=':', linewidth=1.5
            ))

            ax.plot([], [], 's', color=color, alpha=0.6,
                    label=f'Pixel ({i}, {j}), r = {r:.2f} deg: {neighbours.sum()} pixels averaged')

        ax.plot(
            to_unit(x0), to_unit(y0), '+', color='black', markersize=12, markeredgewidth=2,
            path_effects=[patheffects.withStroke(linewidth=4, foreground='white')], label='Model center'
        )

        # Fitted spatial distribution, as its sigma ellipse.
        params = camera_response_params(result)
        if on_counts:
            e = np.sqrt(self.energy_edges[energy_bin_id] * self.energy_edges[energy_bin_id + 1]).to(u.TeV)
            ellipses = [(e, '-')]
        else:
            ellipses = [(params['e_lo'] * u.TeV, '-'), (params['e_hi'] * u.TeV, '-.')]

        single = getattr(result, 'single_bin_fits', {}).get(energy_bin_id) if on_counts else None

        for e, linestyle in ellipses:
            e_tev = e.to_value(u.TeV)
            e_range = (params['e_lo'], params['e_hi'])
            if single is not None:
                # Bin fit on its own, see fit_single_bins().
                sigma, p = single['sigma'], single['p']
                ecc, phi = ellipticity_to_ecc_phi(single['e1'], single['e2'])
                x0e, y0e = single['x0'] * u.deg, single['y0'] * u.deg
            else:
                sigma = energy_saturating(e_tev, params['sigma_lo'], params['sigma_hi'], *e_range, params['sigma_sat'])
                p = energy_power_law(e_tev, params['p_lo'], params['p_hi'], *e_range, clip=True)
                ecc, phi = ellipticity_to_ecc_phi(
                    energy_log_linear(e_tev, params['e1_lo'], params['e1_hi'], *e_range, clip=True),
                    energy_log_linear(e_tev, params['e2_lo'], params['e2_hi'], *e_range, clip=True),
                )
                x0e, y0e = self._model_center(result, e_tev)
            semi_major = sigma / np.sqrt(1 - ecc) * u.deg
            semi_minor = sigma * np.sqrt(1 - ecc) * u.deg
            ax.add_patch(Ellipse(
                (to_unit(x0e), to_unit(y0e)), 2*to_unit(semi_major), 2*to_unit(semi_minor),
                angle=np.degrees(phi),
                fill=False, edgecolor=shape_color, linestyle=linestyle, linewidth=1.8
            ))
            ax.plot([], [], linestyle, color=shape_color,
                    label=f'Sigma ellipse at {e:.2f} ({semi_major:.2f} x {semi_minor:.2f}, '
                          f'phi = {phi:+.2f} rad, p = {p:.2f})')

        ax.plot([], [], '--', color='gray', label=f'|dphi| <= {max_dphi.to_value(u.deg):.3g} deg, '
                                                    f'|dr| <= {max_dr.to_value(u.deg):.3g} deg (at {e_limits.to(u.TeV):.3g})')
        ax.plot([], [], ':', color='gray', label=f'distance <= {max_dist.to_value(u.deg):.3g} deg')

        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        ax.set_aspect('equal')
        ax.set_xlabel(f'X [{ax_unit}]')
        ax.set_ylabel(f'Y [{ax_unit}]')
        if on_counts:
            emin = self.energy_edges[energy_bin_id]
            emax = self.energy_edges[energy_bin_id + 1]
            ax.set_title(f'Counts and smoothing neighbourhoods\nEnergy bin {energy_bin_id}: [{emin:.2f}, {emax:.2f}]')
        else:
            ax.set_title('Smoothing neighbourhoods')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize='small')
        fig.tight_layout()

        return fig, ax

    def plot_fit_check(self, result, energy_bin_id=0, smoothing=None, residual_min_width=0.3*u.deg, ax_unit='deg', cmap='viridis', prior_strength=10.0, val_unit='1 / (s deg2)'):
        """
        Plot a spatial comparison between the observed rate and the
        Bayesian posterior rate (see posterior_rate()) -- the
        camera_response() fit (see fit_camera_response()) used as the
        prior, updated bin-by-bin against the observed data -- for one
        energy bin, as a visual check of the fit quality.

        Produces a 2x2 grid of panels: the observed rate, the posterior
        rate, the smoothed posterior rate (see
        smoothed_posterior_rate()) and the residuals, expressed as the
        relative residual (data - smoothed) / smoothed in counts (or
        (data - posterior) / posterior if no smoothing is given). The
        rates are shown per unit solid angle, each map scaled from 0 to
        its own maximum. The residual is computed and shown
        on a coarser grid, merging consecutive pixels (see
        merged_bin_starts()) narrower than residual_min_width, summing
        the counts of the unmasked pixels only. Each panel has the
        aspect ratio of the x/y ranges.
        Masked pixels (see mask, mask_half(), mask_region()) are left
        blank in the data and residual panels (there is no trustworthy
        data to show or check there), but the posterior panels show the
        actual posterior value there too (the prior, unaffected by
        data) -- the same rate written out for that pixel by
        pybkgmodel.processing.BkgMakerBase.write_maps().

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts.
        energy_bin_id: int
            Energy bin to plot.
        smoothing: dict, optional
            Keyword arguments (max_dphi, max_dr, max_dist) of
            smoothed_posterior_rate() for the smoothed panel, which
            is also used as the model of the residual panel. If None
            (default), that panel is left empty and the residual is
            computed against the unsmoothed posterior.
        residual_min_width: astropy.units.Quantity
            Minimal pixel width of the residual panel. Pixels narrower
            than this are merged before computing the residual.
            Defaults to 0.3 deg.
        ax_unit: str
            Unit to use for the x/y axes.
        cmap: str
            Colormap to use for the data/posterior rate maps.
        prior_strength: float or array_like
            Prior weight passed to posterior_counts() / posterior_rate().
            Defaults to 10.0.
        val_unit: str
            Unit to use for the data/posterior rate maps (rate per
            solid angle).

        Returns
        -------
        fig, axes: matplotlib figure and 2x2 array of axes
            ((data, posterior), (smoothed posterior, residual)).
        """
        mask = self.mask
        data_counts = self.counts[energy_bin_id]
        pixel_area = self.pixel_area

        if smoothing is not None:
            smoothed_rate = self.smoothed_posterior_rate(
                result, prior_strength=prior_strength, energy_bins=[energy_bin_id], **smoothing
            )[0]
            model_counts = (smoothed_rate * self.raw_exposure).decompose().value
            residual_name = 'smoothed'
        else:
            model_counts = self.posterior_counts(result, prior_strength=prior_strength)[energy_bin_id]
            residual_name = 'posterior'

        # Residual on a coarser grid, summing over the unmasked pixels
        # of each merged pixel.
        xstarts = merged_bin_starts(self.xedges, residual_min_width)
        ystarts = merged_bin_starts(self.yedges, residual_min_width)

        def merge(values):
            return np.add.reduceat(np.add.reduceat(values, xstarts, axis=0), ystarts, axis=1)

        res_xedges = self.xedges[np.append(xstarts, len(self.xedges) - 1)]
        res_yedges = self.yedges[np.append(ystarts, len(self.yedges) - 1)]
        merged_data = merge(np.where(mask, data_counts, 0))
        merged_model = merge(np.where(mask, model_counts, 0))
        merged_valid = (merge(mask.astype(int)) > 0) & (merged_model > 0)

        pull = np.full(merged_data.shape, np.nan)
        pull[merged_valid] = (merged_data[merged_valid] - merged_model[merged_valid]) / merged_model[merged_valid]

        with np.errstate(invalid='ignore', divide='ignore'):
            data = (data_counts / self.raw_exposure / pixel_area).to_value(val_unit)
        data[~mask] = np.nan
        model = (self.posterior_rate(result, prior_strength=prior_strength)[energy_bin_id] / pixel_area).to_value(val_unit)

        panels = [
            (self.xedges, self.yedges, data, f'Data rate [{val_unit}]', dict(vmin=0, vmax=np.nanmax(data), cmap=cmap)),
            (self.xedges, self.yedges, model, f'Posterior rate [{val_unit}]', dict(vmin=0, vmax=model.max(), cmap=cmap)),
        ]

        if smoothing is not None:
            smoothed = (smoothed_rate / pixel_area).to_value(val_unit)
            fraction = smoothing.get('data_peak_fraction')
            data_only = fraction is not None and self.data_only_bins(result, fraction)[energy_bin_id]
            single_fit = energy_bin_id in getattr(result, 'single_bin_fits', {})
            if data_only:
                smoothed_title = 'Smoothed data rate (no fit)'
            elif single_fit:
                smoothed_title = 'Smoothed posterior rate (single-bin fit)'
            else:
                smoothed_title = 'Smoothed posterior rate'
            panels.append((
                self.xedges, self.yedges, smoothed, f'{smoothed_title} [{val_unit}]',
                dict(vmin=0, vmax=smoothed.max(), cmap=cmap)
            ))
        else:
            panels.append(None)

        res_width = np.diff(res_xedges).min().to_value(ax_unit)
        panels.append((
            res_xedges, res_yedges, pull,
            f'Residual ({res_width:.2g} {ax_unit} bins)\n(data - {residual_name}) / {residual_name}',
            dict(vmin=-2, vmax=2, cmap='coolwarm')
        ))

        fig, axes = pyplot.subplots(2, 2, figsize=(12, 10))

        for ax, panel in zip(axes.ravel(), panels):
            if panel is None:
                ax.set_visible(False)
                continue

            xedges, yedges, val, title, kwargs = panel
            im = ax.pcolormesh(xedges.to_value(ax_unit), yedges.to_value(ax_unit), val.transpose(), **kwargs)
            ax.set_title(title)
            ax.set_xlabel(f'X [{ax_unit}]')
            ax.set_ylabel(f'Y [{ax_unit}]')
            ax.set_aspect('equal')
            # Colorbar axis of the same height as the (equal-aspect) panel.
            cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
            fig.colorbar(im, cax=cax)

        emin = self.energy_edges[energy_bin_id]
        emax = self.energy_edges[energy_bin_id + 1]
        fig.suptitle(f'Energy bin {energy_bin_id}: [{emin:.2f}, {emax:.2f}]')
        fig.tight_layout()

        return fig, axes

    def plot_radial_profile(self, result, energy_bin_id, smoothing=None, prior_strength=10.0, dr=0.2*u.deg,
                            core_radius=0.5*u.deg, ax_unit='deg'):
        """
        Compare the peakedness of the data and of the model in one
        energy bin: counts per pixel in rings around the model center of
        that bin (the smoothing center, see _bin_centers()), for the
        data, the fitted model (result.model_counts) and, if smoothing
        is given, the final smoothed posterior (see
        smoothed_posterior_rate()), with the data / model ratio below.
        Only the unmasked pixels are used.

        The title gives the counts within core_radius of the center, for
        the data and the fitted model, with the two-sided Poisson
        probability of the observed count given the model.

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response() (after fit_single_bins() if
            it was run).
        energy_bin_id: int
            Energy bin to plot.
        smoothing: dict, optional
            Keyword arguments of smoothed_posterior_rate() (max_dphi,
            max_dr, max_dist, data_peak_fraction) to also show the final
            smoothed posterior. If None (default), it is not shown.
        prior_strength: float or array_like
            Prior weight, see posterior_counts().
        dr: astropy.units.Quantity
            Ring width. Defaults to 0.2 deg.
        core_radius: astropy.units.Quantity
            Radius of the central region tested in the title. Defaults
            to 0.5 deg.
        ax_unit: str
            Unit to use for the radius axis.

        Returns
        -------
        fig, (ax_profile, ax_ratio): matplotlib figure and axes
        """
        x0, y0 = self._bin_centers(result, [energy_bin_id])
        x0, y0 = x0[0].to_value(u.deg), y0[0].to_value(u.deg)
        x = ((self.xedges[1:] + self.xedges[:-1]) / 2).to_value(u.deg)
        y = ((self.yedges[1:] + self.yedges[:-1]) / 2).to_value(u.deg)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        r = np.hypot(xx - x0, yy - y0)
        mask = self.mask

        data = self.raw_counts[energy_bin_id] * mask
        curves = {'Fitted model': result.model_counts[energy_bin_id] * mask}
        if smoothing is not None:
            smoothed = self.smoothed_posterior_rate(
                result, prior_strength=prior_strength, energy_bins=[energy_bin_id], **smoothing
            )[0]
            curves['Smoothed posterior (final)'] = (smoothed * self.raw_exposure).decompose().value * mask

        step = dr.to_value(u.deg)
        edges = np.arange(0, r[mask].max() + step, step) if mask.any() else np.array([0, step])
        centers = (edges[1:] + edges[:-1]) / 2
        rings = [mask & (r >= a) & (r < b) for a, b in zip(edges[:-1], edges[1:])]
        n_pix = np.array([ring.sum() for ring in rings])
        valid = n_pix > 0

        def per_pixel(values):
            return np.array([values[ring].sum() for ring in rings])[valid] / n_pix[valid]

        d = per_pixel(data)
        d_err = np.sqrt(np.array([data[ring].sum() for ring in rings])[valid]) / n_pix[valid]
        rc = (centers[valid] * u.deg).to_value(ax_unit)

        fig, (ax_profile, ax_ratio) = pyplot.subplots(
            2, 1, sharex=True, figsize=(6, 6), gridspec_kw=dict(height_ratios=[3, 1.3])
        )
        ax_profile.errorbar(rc, d, yerr=d_err, fmt='o', color='k', label='Data')
        for (label, values), color in zip(curves.items(), ('C1', 'C0')):
            m = per_pixel(values)
            ax_profile.plot(rc, m, '-', color=color, label=label)
            with np.errstate(invalid='ignore', divide='ignore'):
                ax_ratio.errorbar(rc, d / m, yerr=d_err / m, fmt='o', color=color, ms=4, label=label)

        core = mask & (r < core_radius.to_value(u.deg))
        n_obs, n_exp = data[core].sum(), curves['Fitted model'][core].sum()
        if n_exp > 0:
            p_value = min(1.0, 2 * min(scipy.stats.poisson.cdf(n_obs, n_exp), scipy.stats.poisson.sf(n_obs - 1, n_exp)))
            core_text = (f'within {core_radius.to_value(u.deg):g} deg: data {n_obs:.0f}, '
                         f'model {n_exp:.1f} (p = {p_value:.2g})')
        else:
            core_text = ''

        emin = self.energy_edges[energy_bin_id]
        emax = self.energy_edges[energy_bin_id + 1]
        ax_profile.set_title(f'Energy bin {energy_bin_id}: [{emin:.2f}, {emax:.2f}]\n{core_text}', fontsize='medium')
        ax_profile.set_yscale('log')
        ax_profile.set_ylabel('Counts per pixel')
        ax_profile.legend(fontsize='small')
        ax_ratio.axhline(1, color='gray', ls='--')
        ax_ratio.set_ylim(0, 2)
        ax_ratio.set_ylabel('Data / model')
        ax_ratio.set_xlabel(f'Distance from the model center [{ax_unit}]')
        fig.tight_layout()

        return fig, (ax_profile, ax_ratio)

    def plot_fit_spectrum(self, result, e_unit='TeV', prior_strength=10.0, val_unit='1/s', data_peak_fraction=None):
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

        Data points in energy bins used in the fit (see
        fit_energy_range()) are filled circles, those of the bins fit on
        their own (see fit_single_bins()) open diamonds, those smoothed
        from the data alone (see data_only_bins()) open squares, and
        the others open circles. Dotted lines mark e_lo
        and e_hi, the energies at which the shape parameters are given
        (see camera_response()).

        Parameters
        ----------
        result: scipy.optimize.OptimizeResult
            Result of fit_camera_response(), must have a model_counts
            attribute of the same shape as CameraImage.counts.
        e_unit: str
            Unit to use for the energy axis.
        prior_strength: float or array_like
            Prior weight passed to posterior_counts() / posterior_rate().
            Defaults to 10.0.
        val_unit: str
            Unit to use for the count-rate axis.
        data_peak_fraction: float, optional
            Low-count energy bins smoothed from the data alone, see
            smoothed_posterior_rate(). Only used to mark them.

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
        with np.errstate(invalid='ignore', divide='ignore'):
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

        fitted, _, _ = self.fit_energy_range()
        params = camera_response_params(result)

        if data_peak_fraction is None:
            data_only = np.zeros_like(fitted)
        else:
            data_only = self.data_only_bins(result, data_peak_fraction)
        single_fit = np.zeros_like(fitted)
        single_fit[list(getattr(result, 'single_bin_fits', {}))] = True

        for sel, fmt, mfc, label in (
            (fitted, 'o', 'k', 'Data (fitted)'),
            (single_fit, 'D', 'none', 'Data (single-bin fit)'),
            (~fitted & ~data_only & ~single_fit, 'o', 'none', 'Data (not fitted)'),
            (data_only, 's', 'none', 'Data (smoothed from data only)'),
        ):
            if sel.any():
                ax_spec.errorbar(e[sel], data[sel], yerr=data_err[sel], fmt=fmt, color='k', mfc=mfc, label=label)
                ax_pull.plot(e[sel], residual[sel], fmt, color='k', mfc=mfc)
        ax_spec.plot(e, model, '-', color='C1', label='Posterior')

        for ax in (ax_spec, ax_pull):
            for name in ('e_lo', 'e_hi'):
                ax.axvline((params[name] * u.TeV).to_value(e_unit), color='gray', ls=':')
        ax_spec.plot([], [], ':', color='gray',
                     label=f"Shape reference energies ({params['e_lo']:.2f}, {params['e_hi']:.1f} TeV)")
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')
        ax_spec.set_xlim(left=0.1)
        ax_spec.set_ylim(bottom=np.min(data[data > 0]) / 2, top=1.5 * data.max())
        ax_spec.set_ylabel(f'Rate [{val_unit}]')
        ax_spec.legend(fontsize="small")

        ax_pull.axhline(0, color='gray', ls='--')
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

    def mask_outside_fov(self, radius=None):
        """Masks all camera pixels whose centers are further than radius
        from the camera center, and restricts the fitted model to the
        field of view (see inside_fov, fit_camera_response()): the model
        is 0 outside of it. The corners of the rectangular binning lie
        outside the camera, where there are (almost) no events; fitting
        them would pull the model's tail down at the camera edge.

        Parameters
        ----------
        radius : astropy.units.Quantity, optional
            Field of view radius. Defaults to the smallest absolute value
            of the x and y bin edge limits, i.e. the largest circle
            around the camera center within the binned area.

        Returns
        -------
        radius : astropy.units.Quantity
            Field of view radius used.
        """
        if radius is None:
            radius = u.Quantity([abs(self.xedges[0]), abs(self.xedges[-1]),
                                 abs(self.yedges[0]), abs(self.yedges[-1])]).min()

        self.fov_radius = radius
        self.mask = self.mask & self.inside_fov

        return radius

    @property
    def inside_fov(self):
        """Whether each camera pixel's center lies within fov_radius of
        the camera center (all True if fov_radius is None), see
        mask_outside_fov()."""
        nx = self.xedges.size - 1
        ny = self.yedges.size - 1
        if self.fov_radius is None:
            return np.ones((nx, ny), dtype=bool)

        x = ((self.xedges[1:] + self.xedges[:-1]) / 2).to_value(u.deg)
        y = ((self.yedges[1:] + self.yedges[:-1]) / 2).to_value(u.deg)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        return np.hypot(xx, yy) <= self.fov_radius.to_value(u.deg)

    def mask_reset(self):
        """Unmasks all camera pixels and clears the field of view
        restriction (see mask_outside_fov()).
        """
        self.mask = np.ones((self.xedges.size - 1, self.yedges.size - 1), dtype=bool)
        self.fov_radius = None


    def plot(self, energy_bin_id=0, ax_unit='deg', val_unit='1/s', **kwargs):
        pyplot.xlabel(f'X [{ax_unit}]')
        pyplot.ylabel(f'Y [{ax_unit}]')
        # Masked pixels (e.g. always inside an exclusion region, with
        # zero exposure) are left blank instead of giving 0/0.
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = (self.counts[energy_bin_id] / self.raw_exposure).to_value(val_unit)
        rate[~self.mask] = np.nan

        pyplot.pcolormesh(
            self.xedges.to(ax_unit).value,
            self.yedges.to(ax_unit).value,
            rate.transpose(),
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
