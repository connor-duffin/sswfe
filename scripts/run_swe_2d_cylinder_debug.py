import h5py
import logging

from tqdm import tqdm
from argparse import ArgumentParser
from swe import ShallowTwo

logging.basicConfig(format='%(asctime)s - %(relativeCreated)d ms - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

mesh_file = "mesh/channel-hole.xdmf"
checkpoint_file = "outputs/swe-cylinder-checkpoint.h5"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": 1,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": False
                 })
swe.setup_checkpoint(checkpoint_file)

t = 0.
nt = 30_001
nt_thin = 100
for i in tqdm(range(nt)):
    # store outputs
    if i % nt_thin == 0:
        swe.checkpoint_save(t)

    t += swe.dt
    try:
        swe.solve()
    except RuntimeError:
        logging.info("failed at iteration %d, saving and exiting", i)
        swe.checkpoint_save()
        break

swe.checkpoint_close()
