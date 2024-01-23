# Methods for generating random distributions to matches the functionality of
# the fitting function. As it is not simple to allow the numpy random number
# generator accept custom processes, all function will take the size as the
# format, and an rng variable to consistent generation.
from typing import Optional

import numpy


def generate_poisson(
    size: int, poisson_mean: float, rng: Optional[numpy.random.Generator] = None
) -> numpy.ndarray:
    """
    Unified interface to generate vanilla possion random numbers
    """
    rng = numpy.random.default_rng() if rng is None else rng
    return rng.poisson(poisson_mean, size=size)


def generate_genpoisson(
    size: int,
    poisson_mean: float,
    poisson_borel: float = 0,
    rng: Optional[numpy.random.Generator] = None,
) -> numpy.ndarray:
    """
    Generating random integers which follows the general poisson distribution
    with branching borel fraction.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    ret = rng.poisson(poisson_mean, size=size)
    assert 0 <= poisson_borel < 1, "Borel branching factor must be in range [0,1)"
    branch = ret.copy()
    while numpy.any(branch > 0):
        # numpy can have a per-variable PDF variable!
        branch = rng.poisson(branch * poisson_borel)
        ret = ret + branch
    return ret


def generate_ap_response_smeared(
    size: int,
    smear: numpy.float64,
    n_ap: numpy.int32,
    beta: numpy.float64,
    rng: Optional[numpy.random.Generator] = None,
):
    """
    Generating the afterpulsing contributions to the primary discharges
    """
    rng = numpy.random.default_rng() if rng is None else rng
    ret = rng.exponential(scale=beta, size=(size, n_ap))
    ret = numpy.sum(ret, axis=-1)
    return ret + rng.normal(loc=0, scale=smear, size=size)


def generate_dark_current_contrib(
    size: int,
    ratio: float,
    gain: float,
    rng: Optional[numpy.random.Generator] = None,
) -> numpy.ndarray:
    """
    Generting random sub-pe contribution generated from dark current. The
    parameter ratio represent the result of integration window/SiPM time-scale.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    t = rng.uniform(-1.6 * ratio, ratio, size=size)  # Getting random start time

    return (
        numpy.where(
            t < 0,  # Different contributions for per trigger and post trigger
            numpy.exp(t) * (1 - numpy.exp(-ratio)),
            1 - numpy.exp((-ratio - t)),
        )
        * gain
    )


# GLOBAL VARIABLE used to set the maxiumum AP response in particle count
_GENERATE_MAX_AP_COUNT = 10


def _generate_ap_contrib(
    n_pe: numpy.ndarray,
    ap_prob: float,
    ap_beta: float,
    rng: Optional[numpy.random.Generator] = None,
) -> numpy.ndarray:
    """
    Given an integer array of the primary discharge count for each event,
    return a randomized number of after pulse counts for each event.
    """
    rng = numpy.random.default_rng() if rng is None else rng
    ap_shape = (len(n_pe), _GENERATE_MAX_AP_COUNT)
    ap_contrib = rng.exponential(
        ap_beta, size=ap_shape
    )  # Generating random ap contributions
    # Number of AP cannot exceed n_pe
    ap_contrib = numpy.where(
        numpy.indices(ap_shape)[1] < n_pe[:, numpy.newaxis], ap_contrib, 0
    )
    # AP only has finite chance of appearing
    ap_contrib = numpy.where(rng.random(size=ap_shape) > ap_prob, 0, ap_contrib)

    # For each event, we only see sum of ap responses
    return numpy.sum(ap_contrib, axis=-1)

    # Constructing the response with no random smearing
    return ap_contrib


def generate_response(
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
    rng = numpy.random.default_rng() if rng is None else rng
    ap_beta = gain if ap_beta == 0 else ap_beta  # Making sure we don't have beta=0

    n_pe = generate_genpoisson(
        size, poisson_mean=poisson_mean, poisson_borel=poisson_borel, rng=rng
    )
    noise = numpy.sqrt(common_noise**2 + n_pe * pixel_noise**2)

    pe_contrib = n_pe * gain
    ap_contrib = _generate_ap_contrib(  # Getting the number
        n_pe,
        ap_prob=ap_prob,
        ap_beta=ap_beta,
        rng=rng,
    )
    dc_contrib = numpy.where(
        rng.random(size=size) < dc_prob,
        generate_dark_current_contrib(
            size, ratio=-numpy.log(dc_res), gain=gain, rng=rng
        ),
        0,
    )

    return rng.normal(loc=pedestal + pe_contrib + ap_contrib + dc_contrib, scale=noise)


def generate_response_no_dark_no_ap(
    size: int,
    pedestal: float,
    gain: float,
    common_noise: float,
    pixel_noise: float,
    poisson_mean: float,
    poisson_borel: float,
    rng: Optional[numpy.random.Generator] = None,
):
    """
    Generating response with no dark current and not after pulse contribution
    """
    print(poisson_borel)
    return generate_response(
        size=size,
        pedestal=pedestal,
        gain=gain,
        common_noise=common_noise,
        pixel_noise=pixel_noise,
        poisson_mean=poisson_mean,
        poisson_borel=poisson_borel,
        ap_prob=0.0,
        ap_beta=gain,
        dc_prob=0.0,
        dc_res=0.1,
        rng=rng,
    )


def generate_response_no_dark(
    size: int,
    pedestal: float,
    gain: float,
    common_noise: float,
    pixel_noise: float,
    poisson_mean: float,
    poisson_borel: float,
    ap_prob: float,
    ap_beta: float,
    rng: Optional[numpy.random.Generator] = None,
):
    return generate_response(
        size=size,
        pedestal=pedestal,
        gain=gain,
        common_noise=common_noise,
        pixel_noise=pixel_noise,
        poisson_mean=poisson_mean,
        poisson_borel=poisson_borel,
        ap_prob=ap_prob,
        ap_beta=ap_beta,
        dc_prob=0.0,
        dc_res=0.1,
        rng=rng,
    )
    pass
