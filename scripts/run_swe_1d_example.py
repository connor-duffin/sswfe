import dolfin as fe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser
from swe import ShallowOne

parser = ArgumentParser()
parser.add_argument("--simulation", default="dam_break")
args = parser.parse_args()

if args.simulation == "dam_break":
    nt = 240
    control = {"nx": 1000, "dt": 0.25, "theta": 1.0, "simulation": "dam_break"}

    u_lim = [0., 3.]
    h_lim = [0., 5.]
elif args.simulation == "tidal_flow":
    nt = 3600
    control = {"nx": 1000, "dt": 2.5, "theta": 1.0, "simulation": "tidal_flow"}

    u_lim = [0., 0.18]
    h_lim = [0., 70]
else:
    print("simulation setting not recognised!")
    raise ValueError

t = 0.
swe = ShallowOne(control)
u_out = np.zeros((nt, swe.x_coords.shape[0]))
h_out = np.zeros((nt, swe.x_coords.shape[0]))

if args.simulation == "tidal_flow":
    h_out += swe.H.compute_vertex_values()

for i in range(nt):
    t += swe.dt
    swe.solve(t)
    u_out[i, :] = swe.du.compute_vertex_values()[:(control["nx"] + 1)]
    h_out[i, :] += swe.du.compute_vertex_values()[(control["nx"] + 1):]

fig, ax = plt.subplots()
ax.set_ylim(u_lim)
line, = ax.plot(swe.x_coords, 0 * u_out[0])


def animate(i):
    line.set_ydata(u_out[i])  # update the data.
    return line,


ani = animation.FuncAnimation(fig,
                              animate,
                              interval=20,
                              blit=True,
                              frames=nt,
                              save_count=50)
writer = animation.FFMpegWriter(fps=24,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save(f"figures/u_{args.simulation}.mp4", writer=writer)
plt.close()

fig, ax = plt.subplots()
ax.set_ylim(h_lim)
line, = ax.plot(swe.x_coords, 0 * h_out[0])


def animate(i):
    line.set_ydata(h_out[i])  # update the data.
    return line,


ani = animation.FuncAnimation(fig,
                              animate,
                              interval=20,
                              blit=True,
                              frames=nt,
                              save_count=50)
writer = animation.FFMpegWriter(fps=24,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save(f"figures/h_{args.simulation}.mp4", writer=writer)
plt.close()

u, h = fe.split(swe.du)
# plot velocity component
fe.plot(u)
plt.ylim(u_lim)
plt.savefig(f"figures/u_{args.simulation}_final.png")
plt.close()

# plot height component
if args.simulation == "tidal_flow":
    fe.plot(swe.H + h)
else:
    fe.plot(h)

plt.ylim(h_lim)
plt.savefig(f"figures/h_{args.simulation}_final.png")
plt.close()
