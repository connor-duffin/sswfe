import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

from tqdm import tqdm
from fenics import set_log_level

# setup
import sys
sys.path.insert(0, "./scripts")
set_log_level(40)

from swe import ShallowOne, ShallowOneLinear

control = {"nx": 500,
           "dt": 1e-2,
           "theta": 0.6,
           "simulation": "immersed_bump"}
swe = ShallowOne(control, {"nu": 1e-6})
swe_damped = ShallowOne(control, {"nu": 1e-2})
swe_linear = ShallowOneLinear(control, {"nu": 0.})

t = 0.
t_final = 5 * 60.
nt = np.int64(t_final / control["dt"])
fig, ax = plt.subplots(1, 1, constrained_layout=True)
for i in tqdm(range(nt)):
    t += swe.dt
    swe.solve(t)
    swe_damped.solve(t)

    if i % 1000 == 0:
        u, h = swe.get_vertex_values()
        u_linear, h_linear = swe_linear.get_vertex_values()
        u_damped, h_damped = swe_damped.get_vertex_values()
        ax.plot(swe.x_coords, h, label=f"{t:.5f}")
        # ax.plot(swe.x_coords, h_linear, "--", )
        ax.plot(swe.x_coords, h_damped, "--", )

plt.show()
