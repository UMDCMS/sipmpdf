"""
pdf.py

TODO: insert description of functions here

"""

import zfit

class DemoSiPMPDF(zfit.pdf.ZPDF):
  _N_OBS = 1
  _PARAMS = [
    'ap_beta', 'ap_prob', 'pedestal', 'gain','poisson_mean','poisson_borel','common_noise','pixel_noise','dc_prob','dc_res'
  ]

  def _unnormalized_pdf(self, x):
    x = zfit.z.unstack_x(x)
    return f.sipm_response(
      x, **{k: self.params[k]
            for k in self._PARAMS})

