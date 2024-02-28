# pynajax 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pynapple-org/pynajax/blob/main/LICENSE)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Welcome to `pynajax`, a GPU accelerated backend for `pynapple` built on top on `JAX`.

## Installation
Run the following `pip` command in your virtual environment.

**For macOS/Linux users:**
 ```bash
 pip install git+https://github.com/pynapple-org/pynajax.git
 ```

**For Windows users:**
 ```
 python -m pip install git+https://github.com/pynapple-org/pynajax.git
 ```

## Basic usage

Once you install `pynajax` in your environment, you can set the `pynapple` backend to `JAX` as follows,

```python
import pynapple as nap

nap.nap_config.set_backend("jax")
```

## Disclaimer

Please note that this package is currently under development. While you can
download and test the functionalities that are already present, please be aware
that syntax and functionality may change before our preliminary release.

