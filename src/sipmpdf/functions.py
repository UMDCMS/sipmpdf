"""

functions.py

Pure-function implementation of the various functions used to construction of
the SiPM response. As ultimately, these functions will need to be compatible
with z-fit, we will need make sure these functions works with tensorflow.
However, since numpy is easier to debug with, the functions here will have a
simple switch between numpy/scipy and tensorflow kernels.

"""
from typing import Optional  # Typing not strictly enforced, using numpy dtype for guiding

import warnings

import numpy as np
import tensorflow as tf
import scipy.special

from .kernel import kernel_switch, expand_shape


def generalized_poisson(k: np.int64,
                        mean: np.float64,
                        borel: np.float64 = 0.0) -> np.float64:
  """Generalized Poisson distribution probability values:

  P(k;m,b) = (m (m + k b)^(k-1) exp(-(m + k b))) / k!

  Parameters
  ----------
  k : numpy.int64
      Number of incidents of interest.
  mean : numpy.float64
      Poisson mean of underlying distribution (m)
  borel : numpy.float64
      Borel-branching parameter (b), by default 0.

  Returns
  -------
  numpy.float64
      The probability of k incidents occurring given mean and branching
      parameter.
  """
  kern = kernel_switch(k, mean, borel)

  # Range sanity checks
  # assert kern.all(k >= 0)
  # assert kern.all(mean > 0)
  # assert kern.all(borel >= 0)
  # Use the loggamma function to ensure numerical stability

  return kern.exp(
    kern.log(mean) \
    + (k - 1) * kern.log(mean + k * borel)  \
    - (mean + k * borel)  \
    - kern.loggamma(k + 1))


def normal(x: np.float64,
           mean: np.float64 = 0.0,
           scale: np.float64 = 1.0) -> np.float64:
  """
  Normal distribution function.

  Parameters
  ----------
  x : numpy.float64
      observable
  mean : numpy.float64, optional
      mean of the underlying normal distribution, by default 0.0
  scale : numpy.float64, optional
      scale of the underlying normal distribution, by default 1.0

  Returns
  -------
  numpy.float64
      The probability density of the observable x
  """
  kern = kernel_switch(x, mean, scale)
  # assert kern.all(scale > 0)
  return 1 / (scale * kern.sqrt(2 * kern.pi)) * kern.exp(-(
    (x - mean) / scale)**2 / 2)


def normal_cdf(x: np.float64,
               mean: np.float64 = 0.0,
               scale: np.float64 = 1.0) -> np.float64:
  """
  Function values of the CDF of the normal distribution

  Parameters
  ----------
  x : numpy.float64
      The observable value
  mean : numpy.float64, optional
      The mean of the underlying normal distribution
  scale : numpy.float64, optional
      The scale of the underlying normal distribution

  Returns
  -------
  numpy.float64
      Function value of the CDF
  """
  kern = kernel_switch(x, mean, scale)
  # Range sanity checks
  # assert kern.all(scale > 1.0)

  return (1 + kern.erf((x - mean) / kern.sqrt(scale))) / 2


@expand_shape
def sipm_response_no_dark_no_ap(x: np.float64,
                                pedestal: np.float64,
                                gain: np.float64,
                                common_noise: np.float64,
                                pixel_noise: np.float64,
                                poisson_mean: np.float64,
                                poisson_borel: np.float64 = 0) -> np.float64:
  """
  Testing function to give sipm response with no dark current or afterpulsing 

  Parameters
  ----------
  x : numpy.float64
      The observable value
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  common_noise : np.float64
      Common factor noise
  pixel_noise : np.float64
      Average pixel gain variation
  poisson_mean : np.float64
      Mean optical power of PE discharges
  poisson_borel : np.float64
      Poisson borel value
  Returns
  -------
  numpy.float64
      SiPM response assuming no dark current and after pulsing contributions
  """

  kern = kernel_switch(x, pedestal, gain, common_noise, pixel_noise,
                       poisson_mean, poisson_borel)
  # Extracting the number of discharges to look for.
  n_max = kern.reduce_max(poisson_mean, axis=None)
  n_max = kern.toint32(kern.rint(n_max + 5 * kern.sqrt(n_max) + 15))

  def extend_arr(x):
    return kern.repeat(x[kern.newaxis, ...], n_max, axis=0)

  # Extending input array structure
  x = extend_arr(x)
  pedestal = extend_arr(pedestal)
  gain = extend_arr(gain)
  common_noise = extend_arr(common_noise)
  pixel_noise = extend_arr(pixel_noise)
  poisson_mean = extend_arr(poisson_mean)
  poisson_borel = extend_arr(poisson_borel)

  # Constructing the discharge count array structure
  n_pe = kern.tofloat64(kern.local_index(poisson_mean, axis=0))

  noise = kern.sqrt(common_noise**2 + n_pe * pixel_noise**2)
  prob_n = generalized_poisson(n_pe, poisson_mean, poisson_borel)
  norm = normal(x, pedestal + gain * n_pe, noise)
  return kern.sum(norm * prob_n, axis=0)


def binomial_prob(x: np.int64, total: np.int64, prob: np.float64) -> np.float64:
  """
  Binomial probability. Implementing as pure numpy function to ensure that
  acceleration is possible later down the line.

  Parameters
  ----------
  x : numpy.int64
      Number of observed events that is selected
  total : numpy.int64
      Total number of observed events.
  prob : numpy.float64
      Underlying probability that events pass selection
  """
  kern = kernel_switch(x, total, prob)
  # Range sanity check
  #assert kern.all(0.0 <= x) & kern.all(0.0 < total)
  #assert kern.all(0.0 <= prob) & kern.all(prob <= 1.0)

  # Implementing using log-exp to ensure numerical stability
  # Where to ensure that x is less than or equal to total, if not return 0
  x = kern.tofloat64(x)
  return kern.where(x<=total,kern.exp(x * kern.log(prob)  \
                  + (total - x) * kern.log(1 - prob) \
                  + kern.loggamma(total + 1) \
                  - kern.loggamma(x + 1)  \
                  - kern.loggamma(total - x + 1)),0)


def ap_response(x: np.float64, n_ap: np.float64, beta: np.float64) -> np.float64:
  """
  Afterpulse response given a fixed number of after pulses occurs.

  Parameters
  ----------
  x : np.float64
      Observation, above the primary discharge peak
  n_ap : np.float64
      number of afterpulses that occur
  beta : np.float64
      Afterpulse timescale factor.
  Returns
  -------
  np.float64
      Probability density of the after pulse response
  """
  kern = kernel_switch(x, n_ap, beta)

  return kern.where(
    x > 0,
    kern.exp(
      kern.log(kern.power(x, n_ap - 1)) - kern.log(kern.power(beta, n_ap)) -
      kern.loggamma(n_ap) - x / beta), 0)


def darkcurrent_response_original(x: np.float64,
                                  gain: np.float64,
                                  resolution: np.float64 = 1e-4) -> np.float64:
  """
  Dark current response without random noise, given a detector resolution factor
  (calculated relative to the gain of the system)

  Parameters
  ----------
  x : np.float64
      Observation
  gain : np.float64
      The gain of the SiPM
  resolution : np.float64, optional
      resolution factor, relative to the SiPM gain factor, by default 1e-4

  Returns
  -------
  np.float64
      Probability density of observing a dark current response
  """
  kern = kernel_switch(x, gain, resolution)

  eps = gain * resolution

  left = 1 / x
  right = 1 / (gain - x)
  norm = 2 * (kern.log(1 - resolution) - kern.log(resolution))
  return kern.where(((x > eps) & (x < (gain - eps))),  #
                    (left + right) / norm, 0)


def darkcurrent_response(x: np.float64,
                         gain: np.float64,
                         resolution: np.float64 = 1e-4) -> np.float64:
  """
  Modified dark current response, as the unmodified dark current response has
  excessively sharp features that can make other numerical tools unstable, the
  modified version attempts smooth out the dark current response by substituting
  out the sharp features at x~pedestal and x~(pedestal+gain) with a normal
  response with a width of ~0.01 resolution (since we don't expect any detector
  to have such low a noise, 0.01 feels like a reasonable cutoff)

  Here we modify the function to preserve the overall normalization and the 0-th
  order smoothness of the function (function is continuous, but derivative is
  not):

  - If x is in the "nominal region" of (x > pedestal + 0.01 * gain) and (x <
    pedestal - 0.01 * gain), we will return the original response function.
  - If x is in the "peaking regions", the response is substituted with a Gaussian that is x50 wider than the resolution factor.
  
  Parameters
  ----------
  x : np.float64
      Observation
  gain : np.float64
      The gain of the SiPM
  resolution : np.float64, optional
      resolution factor, relative to the SiPM gain factor, by default 1e-4

  Returns
  -------
  np.float64
      Smoothed out probability density of observing a dark current response
  
  """
  kern = kernel_switch(x, gain, resolution)
  coarse_res = resolution * 50
  coarse_res = kern.where(coarse_res > kern.tofloat64(0.01),
                          kern.tofloat64(0.01), coarse_res)
  eps = gain * resolution
  coarse = gain * coarse_res

  nominal = kern.where(((x > coarse) & (x < (gain - coarse))),  #
                       darkcurrent_response_original(x, gain, resolution), 0.0)

  outer = normal(x, 0, coarse) + normal(x, gain, coarse)
  outer_norm = (kern.log(coarse_res) - kern.log(resolution)) / 2.0 / (
    kern.log(1 - resolution) - kern.log(resolution))
  return nominal + outer * outer_norm


def ap_response_smeared(x: np.float64, smear: np.float64, n_ap: np.int64,
                        beta: np.float64) -> np.float64:
  """
  Afterpulse response, assuming a fixed number of afterpulses, taking into account random noise effects.
  
  Parameters
  ----------
  x : np.float64
      Observation
  smear: np.float64
      Scale of random noise
  n_ap : np.float64
      Number of afterpulses that occur
  beta : np.float64
      Afterpulse timescale factor.
  Returns
  -------
  np.float64
      Probability density of the after pulse response
  """
  kern = kernel_switch(x, smear, n_ap, beta)
  return kern.where(
    n_ap <= 1.5,
    kern.exp(-x / beta) *
    (1 + kern.erf(x / (smear * kern.sqrt(kern.tofloat64(2))))) / (2 * beta),
    ap_response(x=x, n_ap=n_ap, beta=beta))


@expand_shape
def darkcurrent_response_smeared(x: np.float64,
                                 smear: np.float64,
                                 gain: np.float64,
                                 resolution: np.float64 = 1e-4) -> np.float64:
  """
  Dark current response with random noise, given a detector resolution factor
  (calculated relative to the gain of the system)

  Parameters
  ----------
  x : np.float64
      Observation
  smear : np.float64
      Scale of random noise
  gain : np.float64
      The gain of the SiPM
  resolution : np.float64, optional
      resolution factor, relative to the SiPM gain factor, by default 1e-4

  Returns
  -------
  np.float64
      Probability density of observing a dark current response
  """

  kern = kernel_switch(x, smear, gain, resolution)

  n_samples = 1024

  #extend arrays
  def extend_arr(x):
    return kern.repeat(x[kern.newaxis, ...], n_samples, axis=0)

  x = extend_arr(x)
  smear = extend_arr(smear)
  gain = extend_arr(gain)
  resolution = extend_arr(resolution)

  delta = 8 * smear / n_samples  # Getting the integration window
  idx = kern.local_index(x, axis=0)
  x_pr = -4 * smear + delta * (idx + 0.5)

  return kern.sum(darkcurrent_response(x - x_pr, gain, resolution) *
                  normal(x_pr, 0, smear) * delta,
                  axis=0)


@expand_shape
def _afterpulsing_summation(x: np.float64, total: np.int64, ap_smear: np.float64,
                            ap_beta: np.float64,
                            ap_prob: np.float64) -> np.float64:
  """
  Summation of the afterpulse response (with random noies) weighted by the binomial distribution for n_ap from 1 to 10
  
  Parameters
  ----------
  x : np.float64
      Observation
  ap_smear: np.float64
      Scale of random noise
  ap_beta : np.float64
      Afterpulse timescale factor.
  Returns
  -------
  np.float64
      Afterpulse response weighted by binomial distribution
  """

  kern = kernel_switch(x, ap_smear, ap_prob, ap_beta, total)
  k = 10

  #extend arrays
  def extend_arr(x):
    return kern.repeat(x[kern.newaxis, ...], k, axis=0)

  x = extend_arr(x)
  ap_smear = extend_arr(ap_smear)
  ap_beta = extend_arr(ap_beta)
  ap_prob = extend_arr(ap_prob)
  total = extend_arr(total)

  idx = kern.local_index(x, axis=0)
  idx += 1

  return kern.sum(
    binomial_prob(x=idx, total=total, prob=ap_prob) *
    ap_response_smeared(x=x, smear=ap_smear, n_ap=idx, beta=ap_beta),
    axis=0)


def _full_afterpulse_response(x: np.float64, total: np.int64,
                              ap_smear: np.float64, ap_beta: np.float64,
                              ap_prob: np.float64, pedestal: np.float64,
                              gain: np.float64,
                              sigma_k: np.float64) -> np.float64:
  """ 
  Summation of the afterpulse response (with random noise) weighted by the binomial distribution for n_ap from 0 to 10
  
  Parameters
  ----------
  x : numpy.float64
      The observable value
  total : numpy.float64
      The total number of P.E.s
  ap_beta : np.float64
      Afterpulse timescale factor.
  ap_prob : np.float64
      Probability of afterpulsing
  ap_smear : np.float64
      Scale of random noise for afterpulses
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  sigma_k :  np.float64
      Scale of the noise for no afterpulses
  Returns
  -------
  numpy.float64
      Full afterpulse response
  """
  kern = kernel_switch(x, total, ap_smear, ap_beta, ap_prob, pedestal, gain,
                       sigma_k)
  return _afterpulsing_summation(
    x=x - (pedestal + total * gain),
    total=total,
    ap_smear=ap_smear,
    ap_beta=ap_beta,
    ap_prob=ap_prob) + binomial_prob(
      x=kern.zeros_like(total), total=total, prob=ap_prob) * normal(
        x=x, mean=pedestal + total * gain, scale=sigma_k)


@expand_shape
def _k_summation(x: np.float64, common_noise: np.float64,
                 pixel_noise: np.float64, ap_beta: np.float64,
                 ap_prob: np.float64, pedestal: np.float64, gain: np.float64,
                 poisson_mean: np.float64,
                 poisson_borel: np.float64) -> np.float64:
  """ 
  Summation in the final SiPM response for number of primary P.E.s k>=1, dark current at k==0 is handled by a different function
  
  Parameters
  ----------
  x : numpy.float64
      The observable value
  common_noise : np.float64
      Common factor noise corresponding to sigma_0
  pixel_noise : np.float64
      Average pixel gain variation corresponding to sigma_1
  ap_beta : np.float64
      Afterpulse timescale factor.
  ap_prob : np.float64
      Probability of afterpulsing
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  poisson_mean : np.float64
      Mean optical power of PE discharges
  poisson_borel : np.float64
      Poisson borel value
  Returns
  -------
  numpy.float64
      SiPM response for k>=1 not accounting for dark current
  """
  kern = kernel_switch(x, common_noise, pixel_noise, ap_prob, ap_beta, pedestal,
                       gain, poisson_mean, poisson_borel)

  k_max = kern.reduce_max(poisson_mean, axis=None)
  k_max = kern.toint32(kern.rint(k_max + 5 * kern.sqrt(k_max) + 15))

  def extend_arr(x):
    return kern.repeat(x[kern.newaxis, ...], k_max, axis=0)

  x = extend_arr(x)
  common_noise = extend_arr(common_noise)
  pixel_noise = extend_arr(pixel_noise)
  ap_beta = extend_arr(ap_beta)
  ap_prob = extend_arr(ap_prob)
  pedestal = extend_arr(pedestal)
  gain = extend_arr(gain)
  poisson_mean = extend_arr(poisson_mean)
  poisson_borel = extend_arr(poisson_borel)

  idk = kern.local_index(x, axis=0) + 1
  sigma_k = kern.sqrt(
    kern.power(common_noise, 2) + idk * kern.power(pixel_noise, 2))

  return kern.sum(
    generalized_poisson(k=idk, mean=poisson_mean, borel=poisson_borel) *
    _full_afterpulse_response(x=x,
                              total=total,
                              ap_smear=common_noise,
                              ap_beta=ap_beta,
                              ap_prob=ap_prob,
                              pedestal=pedestal,
                              gain=gain,
                              sigma_k=sigma_k),
    axis=0)


def sipm_response(x: np.float64,
                  pedestal: np.float64,
                  gain: np.float64,
                  common_noise: np.float64,
                  pixel_noise: np.float64,
                  poisson_mean: np.float64,
                  poisson_borel: np.float64,
                  ap_prob: np.float64,
                  ap_beta: np.float64,
                  dc_prob: np.float64,
                  dc_res: np.float64 = 1e-4) -> np.float64:
  """
  SiPM lowlight response probability density function
  
  Parameters
  ----------
  x : numpy.float64
      The observable value
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  common_noise : np.float64
      Common factor noise corresponding to sigma_0
  pixel_noise : np.float64
      Average pixel gain variation corresponding to sigma_1
  poisson_mean : np.float64
      Mean optical power of PE discharges
  poisson_borel : np.float64
      Poisson borel value
  ap_beta : np.float64
      Afterpulse timescale factor.
  ap_prob : np.float64
      Probability of afterpulsing
  dc_prob : np.float64
      Probability of dark current
  dc_res : np.float64, optional
      resolution factor, relative to the SiPM gain factor, by default 1e-4
  Returns
  -------
  numpy.float64
      SiPM response probability density function as a function of x
  """

  kern = kernel_switch(x, pedestal, gain, common_noise, pixel_noise,
                       poisson_mean, poisson_borel, ap_prob, ap_beta, dc_prob,
                       dc_res)
  dc_smear = kern.sqrt(common_noise**2 + pixel_noise**2)

  no_pe_discharge_response = generalized_poisson(
    k=kern.zeros_like(x), mean=poisson_mean,
    borel=poisson_borel) * (1 - dc_prob) * normal(
      x=x, mean=pedestal,
      scale=common_noise) + dc_prob * darkcurrent_response_smeared(
        x=x, smear=dc_smear, gain=gain, resolution=dc_res)
  pe_discharge_response = _k_summation(x=x,
                                       common_noise=common_noise,
                                       pixel_noise=pixel_noise,
                                       ap_beta=ap_beta,
                                       ap_prob=ap_prob,
                                       pedestal=pedestal,
                                       gain=gain,
                                       poisson_mean=poisson_mean,
                                       poisson_borel=poisson_borel)

  return no_pe_discharge_response + pe_discharge_response


def sipm_response_no_dark(x: np.float64,
                          pedestal: np.float64,
                          gain: np.float64,
                          common_noise: np.float64,
                          pixel_noise: np.float64,
                          poisson_mean: np.float64,
                          poisson_borel: np.float64,
                          ap_prob: np.float64,
                          ap_beta: np.float64,
                          dc_res: np.float64 = 1e-4) -> np.float64:
  """
  SiPM low light probability density function, excluding dark current effects.
  
  Parameters
  ----------
  x : numpy.float64
      The observable value
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  common_noise : np.float64
      Common factor noise corresponding to sigma_0
  pixel_noise : np.float64
      Average pixel gain variation corresponding to sigma_1
  poisson_mean : np.float64
      Mean optical power of PE discharges
  poisson_borel : np.float64
      Poisson borel value
  ap_beta : np.float64
      Afterpulse timescale factor.
  ap_prob : np.float64
      Probability of afterpulsing
  Returns
  -------
  numpy.float64
      SiPM response without dark current probability density function as a function of x
  """
  kern = kernel_switch(x, pedestal, gain, common_noise, pixel_noise,
                       poisson_mean, poisson_borel, ap_prob, ap_beta)

  no_pe_discharge_response = generalized_poisson(
    k=kern.zeros_like(x), mean=poisson_mean, borel=poisson_borel) * normal(
      x=x, mean=pedestal, scale=common_noise)
  pe_discharge_response = _k_summation(x=x,
                                       common_noise=common_noise,
                                       pixel_noise=pixel_noise,
                                       ap_beta=ap_beta,
                                       ap_prob=ap_prob,
                                       pedestal=pedestal,
                                       gain=gain,
                                       poisson_mean=poisson_mean,
                                       poisson_borel=poisson_borel)

  return no_pe_discharge_response + pe_discharge_response
