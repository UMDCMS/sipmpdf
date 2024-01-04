# Methods for generating random distributions to matches the functionality of
# the fitting function. As it is not simple to allow the numpy random number
# generator accept custom processes, all function will take the size as the
# format, and an rng variable to consistent generation.
from typing import Optional

import awkward
import numpy


def generate_poisson(
    size: int, mu: float, rng: Optional[numpy.random.Generator] = None
) -> numpy.ndarray:
    """
    Unified interface to generate vanilla possion random numbers
    """
    rng = numpy.random.default_rng() if rng is None else rng
    return rng.poisson(mu, size=size)


def generate_genpoisson(
    size: int, mu: float, borel: float = 0, rng: Optional[numpy.random.Generator] = None
) -> numpy.ndarray:
    """
    Generating random integers which follows the general poisson distribution
    with branching borel fraction.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    ret = rng.poisson(mu, size=size)
    assert 0 <= borel < 1, "Borel branching factor must be in range [0,1)"
    branch = ret.copy()
    while numpy.any(branch > 0):
        # numpy can have a per-variable PDF variable!
        branch = rng.poisson(branch * borel)
        ret = ret + branch
    return ret


def _generate_dark_current_contrib(
    size: int, ratio: float, rng: Optional[numpy.random.Generator] = None
) -> numpy.ndarray:
    """
    Generting random sub-pe contribution generated from dark current. The
    parameter ratio represent the result of integration window/SiPM time-scale.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    t = rng.uniform(-ratio, ratio, size=size)  # Getting random start time

    return numpy.where(
        t < 0,  # Different contributions for per trigger and post trigger
        numpy.exp(t) * (1 - numpy.exp(-ratio)),
        1 - numpy.exp((-ratio - t)),
    )


def _generate_ap_contrib(
    n_pe: numpy.ndarray,
    ap_prob: float,
    beta: float,
    common_noise: float,
    pixel_noise: float,
    rng: Optional[numpy.random.Generator] = None,
) -> numpy.ndarray:
    """
    Given an integer array of the primary discharge count for each event,
    return a randomized number of after pulse counts for each event.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    n_ap = rng.binomial(n_pe, p=ap_prob)  # Getting number of afterpulses
    n_ap_total = numpy.sum(n_ap)
    ap_contrib = rng.exponential(beta, size=n_ap_total)  # For each po
    noise = rng.normal(loc=common_noise, scale=pixel_noise, size=n_ap_total)
    ap_contrib = ap_contrib + rng.normal(  # Injecting noise with randomized width
        loc=0, scale=numpy.where(noise > 0, noise, 0), size=n_ap_total
    )
    ap_contrib = awkward.unflatten(ap_contrib, n_ap)  # Folding back to number count
    return awkward.sum(ap_contrib, axis=-1)


def generate_readout(
    size: int,
    pedestal: numpy.float64,
    gain: numpy.float64,
    common_noise: numpy.float64,
    poisson_mean: numpy.float64,
    pixel_noise: numpy.float64 = 0,
    poisson_borel: numpy.float64 = 0,
    ap_prob: numpy.float64 = 0,
    ap_beta: numpy.float64 = 0,
    dc_prob: numpy.float64 = 0,
    dc_res: numpy.float64 = 1e-4,
    rng: Optional[numpy.random.Generator] = None,
):
    """
    Generating fake readout of `size` events (assuming no saturation effects)
    """
    # Some additional parsing
    ap_beta = gain if ap_beta == 0 else ap_beta
    rng = numpy.random.default_rng() if rng is None else rng

    n_pe = generate_genpoisson(size, poisson_mean, poisson_mean, rng)
    pe_contrib = rng.normal(
        loc=n_pe * gain, scale=numpy.sqrt(common_noise**2 + n_pe * pixel_noise**2)
    )
    ap_contrib = _generate_ap_contrib(  # Getting the number
        size,
        ap_prob=ap_prob,
        beta=ap_beta,
        common_noise=common_noise,
        pixel_noise=pixel_noise,
        rng=rng,
    )
    dc_contrib = _generate_dark_current_contrib(
        size, ratio=numpy.log(dc_res), rng=rng
    ) + rng.random(loc=0, scale=numpy.sqrt(common_noise**2 + pixel_noise**2))

    return pe_contrib + ap_contrib + dc_contrib

