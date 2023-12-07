# SSWFE: Statistical Shallow Water via Finite Elements

This repo contains the working code for a collaborative project between the University of Cambridge [Computational Statistics and Machine Learning Group](https://csml-cam.github.io/) in the Engineering Department and the University of Western Australia's [TIDE research hub](https://tide.edu.au/).

This repo is currently hosted here to showcase the model development with `FEniCS`, which focuses on posterior computation using various nonlinear filtering algorithms, to estimate the filtering posterior $p(u_n | y_{1:n})$. This posterior is computed using either the low-rank Extended Kalman filter (LR-ExKF), and the ensemble Kalman filter. As the code stands now, both tend to be unstable and need to be user-tuned.

This setup is that of flow past a cylinder within a symmetric rectangular domain, with an Dirichlet inflow boundary condition and Flather outflow boundary condition. Currently the FEM model is implemented using either an *IMEX* scheme, or, a $\theta$-method, with $P2$ elements on the velocity space and $P1$ elements on the surface heights.

## Setup and installation

The `scripts` directory contains the necessary command line scripts to general the mesh and run the model. The `swfe` directory contains a Python package, which can be installed using an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) via running the following from this directory:

```
pip install -e ./swfe/
```

Once installed unit tests can be run via

```
python3 -m pytest swfe/tests
```

## Data

Unfortunately the data for this project is not publicly available, nor is it likely to be for the timebeing. A placeholder directory is given in `data/` in case you wanted to simulate some fake data to try the models.
