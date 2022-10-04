import h5py
import logging

import fenics as fe
import numpy as np

from tqdm import tqdm
from swe import ShallowOne
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
fe.set_log_level(40)


parser = ArgumentParser()
# sim settings
parser.add_argument("--output_file", type=str)
parser.add_argument("--nu", type=float)
# numerical settings
parser.add_argument("--nx", type=int)
parser.add_argument("--dt", type=float)
parser.add_argument("--n_cycles", type=float)
parser.add_argument("--nt_save", type=int)
args = parser.parse_args()

# args.output_file = "outputs/test.h5"
# args.nu = 0.1
# args.n_cycles = 0.5  # 50

# args.nx = 400
# args.dt = 4.
# args.nt_save = 100

# set the observation system (86_400 / 2 => half a day)
NT = np.int64((args.n_cycles * 86_400 / 2) / args.dt)

logger.info("storing outputs in %s, at every %d timesteps", args.output_file,
            args.nt_save)

control = {"nx": args.nx,
           "dt": args.dt,
           "theta": 1.0,
           "simulation": "tidal_flow"}
params = {"nu": args.nu}
swe = ShallowOne(control, params)
swe.setup_checkpoint(args.output_file)

attrs = swe.checkpoint.attributes("/")
attrs["nu"] = args.nu
attrs["dt"] = args.dt

t = 0.
for i in tqdm(range(NT)):
    t += swe.dt
    try:
        swe.solve(t)
        if i % args.nt_save == 0:
            swe.checkpoint_save(t)
    except RuntimeError:
        logger.error("SWE, nu = %.5f failed at t= %.5f, exiting", args.nu, t)
        swe.checkpoint_save(t)
        break

swe.checkpoint_close()
