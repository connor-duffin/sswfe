import logging
import xarray as xr
import numpy as np

from argparse import ArgumentParser
from fenics import set_log_level
from swe import ShallowOne
from tqdm import tqdm

set_log_level(40)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

settings = dict(nx=500, dt=0.01, theta=0.6, nu=1e-6)
sigma_y = 5e-4
t_final = 300.

swe_dgp = ShallowOne(
    control=dict(nx=settings["nx"],
                 dt=settings["dt"],
                 theta=settings["theta"],
                 simulation="immersed_bump"),
    params=dict(nu=settings["nu"]))

# set the observation system
nt = np.int64(t_final / settings["dt"])
t_grid = np.linspace(0., t_final, nt + 1)

# store outputs
h_obs = np.zeros((nt + 1, settings["nx"] + 1))  # include step for final time
h_obs[0, :] = swe_dgp.du.compute_vertex_values()[(settings["nx"] + 1):]

t = 0.
logger.info("starting SWE run")
for i in tqdm(range(nt)):
    t += swe_dgp.dt
    swe_dgp.solve(t)
    h_obs[i + 1, :] = swe_dgp.du.compute_vertex_values()[(settings["nx"] + 1):]
    h_obs[i + 1, :] += sigma_y * np.random.normal(size=h_obs[i + 1, :].shape)

# HACK(connor): autoconvert to dataset
out = xr.DataArray(data=h_obs,
                   coords=dict(t=t_grid, x=swe_dgp.x_coords.flatten()),
                   attrs=settings,
                   name="h").to_dataset()

logger.info("storing outputs into %s", args.output_file)
out.to_netcdf(args.output_file)
