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
  lamb : numpy.float64
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
  """Function values of the CDF of the normal distribution

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
def normal_smeared_poisson(x: np.float64,
                           pedestal: np.float64,
                           gain: np.float64,
                           common_noise: np.float64,
                           pixel_noise: np.float64,
                           poisson_mean: np.float64,
                           poisson_borel: np.float64 = 0) -> np.float64:
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
  return kern.exp(x * kern.log(prob)  \
                  + (total - x) * kern.log(1 - prob) \
                  + kern.loggamma(total + 1) \
                  - kern.loggamma(x + 1)  \
                  - kern.loggamma(total - x + 1))


def ap_response(x: np.float64, n_ap: np.int64, beta: np.float64) -> np.float64:
  """
  Afterpulse response given a fixed number of after pulses occurs. (with random
  noise effects)

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
  kern = kernel_switch(x, n_np, beta)
  pass


def darkcurrent_response(x: np.float64,
                         pedestal: np.float64,
                         gain: np.float64,
                         resolution: np.float64 = 1e-4) -> np.float64:
  """
  Dark current response without random noise, given a detector resolution factor
  (calculated relative to the gain of the system)

  Parameters
  ----------
  x : np.float64
      Observation
  pedestal : np.float64
      The pedestal value
  gain : np.float64
      The gain of the SiPM
  resolution : np.float64, optional
      resolution factor, relative to the SiPM gain factor, by default 1e-4

  Returns
  -------
  np.float64
      Probability density of observing a dark current response of
  """
  kern = kernel_switch(x, pedestal, gain, resolution)

  eps = gain * resolution
  lo = pedestal
  up = pedestal + gain
  
  return kern.where((x > (lo + eps)) & (x < (up - eps)),
                    ((1 / (x - lo)) + (1 / (up - x))) / (2 * kern.log((up - lo - eps) / eps)), 0)


def ap_response_smeared(x: np.float64, smear: np.float64, n_ap: np.int64,
                        beta: np.float64) -> np.float64:
  kern = kernel_switch(x, smear, n_np, beta)
  pass


def darkcurrent_response_smeared(x: np.float64,
                                 smear: np.float64,
                                 pedestal: np.float64,
                                 gain: np.float64,
                                 resolution: np.float64 = 1e-4) -> np.float64:
  kern = kernel_switch(x, smear, pedestal, gain, resolution)

  """
  notes for Grace:
  inside summation: darkcurrent_response(n*delta,pedestal,gain,resolution)*normal(x-n*delta,0,smear)*delta """

  n_max=256 #change later to 256, just for testing purposes
  delta=(3*smear)/n_max #confirm this is the right measurement
  input_count=len(x)

  #extend arrays
  def extend_arr(x):
    return kern.repeat(x[kern.newaxis, ...], n_max, axis=0)
  x=extend_arr(x)
  smear=extend_arr(smear)
  pedestal=extend_arr(pedestal)
  gain=extend_arr(gain)
  resolution=extend_arr(resolution)
  
  #do math
  narray=kern.indices((n_max,input_count))[0] #indices go from 0 to n_max-1
  result=darkcurrent_response(narray*delta,pedestal,gain,resolution)*delta*normal(x-narray*delta,0,smear)
  return kern.sum(result, axis=0)

def sipm_response(
  x: np.float64,
  pedestal: np.float64,
  gain: np.float64,
  common_noise: np.float64,
  pixel_noise: np.float64,
  poisson_mean: np.float64,
  poisson_borel: np.float64,
  ap_prob: np.float64,
  ap_beta: np.float64,
) -> np.float64:
  """
  Main function for the general SiPM low light response probability function.
  Using numba for calculation acceleration (the function is difficult to express
  in pure array syntax)
  """
  pass
