# SiPM PDF implementation

This is a python implementation of the SiPM low light response function as
described in [this paper (arXiv:1609.01181)][sipm_response]. Function
implemented here can be used together with [`zfit`][zfit] for high-performace
fitting for SiPM response and property analysis.

As this package depends on `zfit`, you will need at least python 3.8 to use this
package. This package can be installed in your python environment like

```bash
python -m pip install git+https://github.com/UMDCMS/sipmpdf@v1.0.0
```

[sipm_response]: https://arxiv.org/pdf/1609.01181.pdf
[zfit]: https://github.com/zfit/zfit
