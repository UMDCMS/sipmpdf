"""
pdf.py

SiPM response functions as a zfit.pdf.ZPDF extended class for user level fits.

"""

import zfit

from . import functions as f


class SiPMResponse_NoDCAP_PDF(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = [
        "pedestal",
        "gain",
        "common_noise",
        "pixel_noise",
        "poisson_mean",
        "poisson_borel",
    ]

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        return f.sipm_response_no_dark_no_ap(
            x, **{k: self.params[k] for k in self._PARAMS}
        )


class SiPMResponse_NoDC_PDF(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = [
        "pedestal",
        "gain",
        "common_noise",
        "pixel_noise",
        "poisson_mean",
        "poisson_borel",
        "ap_prob",
        "ap_beta",
    ]

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        return f.sipm_response_no_dark(x, **{k: self.params[k] for k in self._PARAMS})


class SiPMResponsePDF(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = [
        "pedestal",
        "gain",
        "common_noise",
        "pixel_noise",
        "poisson_mean",
        "poisson_borel",
        "ap_prob",
        "ap_beta",
        "dc_prob",
        "dc_res",
    ]

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        return f.sipm_response(x, **{k: self.params[k] for k in self._PARAMS})
