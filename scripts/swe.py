""" Solve the Shallow-water equations in non-conservative form. """
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser


class PiecewiseIC(fe.UserExpression):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def eval(self, values, x):
        if x[0] < self.L + fe.DOLFIN_EPS:
            values[0] = 5.
        else:
            values[0] = 0.


class ShallowOne:
    def __init__(self, control):
        # settings: L, nu, C
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.simulation = control["simulation"]

        if self.simulation == "dam_break":
            self.L = 2000
            self.nu = 1.0
            self.C = 0.
        elif self.simulation == "tidal_flow":
            self.L = 14_000
            self.nu = 1.0
            self.C = 0.
        else:
            print("simulation setup not recognised")
            raise ValueError

        # setup mesh and function spaces
        self.mesh = fe.IntervalMesh(self.nx, 0., self.L)
        self.x = fe.SpatialCoordinate(self.mesh)
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        U = fe.FiniteElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        self.x_coords = self.mesh.coordinates()
        self.x_dofs = self.W.tabulate_dof_coordinates()
        self.u_dofs = self.W.sub(0).dofmap().dofs()
        self.h_dofs = self.W.sub(1).dofmap().dofs()

        self.x_dofs_u = self.x_dofs[self.u_dofs, :]
        self.x_dofs_h = self.x_dofs[self.h_dofs, :]

        # HACK: introduce new function space to construct interpolant
        V = fe.FunctionSpace(self.mesh, "CG", 1)

        if self.simulation == "dam_break":
            self.H = fe.Constant(5.0)

            p = PiecewiseIC(self.L // 2)
            ic = fe.interpolate(p, V)
        elif self.simulation == "tidal_flow":
            H = fe.Expression(
                "50.5 - 40 * x[0] / L - 10 * sin(pi * (4 * x[0] / L - 0.5))",
                L=self.L,
                degree=2)
            self.H = fe.interpolate(H, V)
        else:
            print("simulation setup not currently enabled")
            raise ValueError

        self.du = fe.Function(W)
        u, h = fe.split(self.du)

        self.du_prev = fe.Function(W)
        u_prev, h_prev = fe.split(self.du_prev)

        v_u, v_h = fe.TestFunctions(W)

        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        C = fe.Constant(self.C)
        dt = fe.Constant(self.dt)

        self.theta = 1.0
        u_mid = self.theta * u + (1 - self.theta) * u_prev
        h_mid = self.theta * h + (1 - self.theta) * h_prev
        u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

        self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx +
                  fe.inner(h - h_prev, v_h) / dt * fe.dx +
                  u_prev * u_mid.dx(0) * v_u * fe.dx +
                  nu * 2 * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx +
                  g * h_mid.dx(0) * v_u * fe.dx + C * u_mag * u_mid * v_u /
                  (self.H + h_mid) * fe.dx -
                  ((self.H + h_mid) * u_mid * v_h.dx(0)) * fe.dx)
        self.J = fe.derivative(self.F, self.du)

        def _right(x, on_boundary):
            return x[0] >= (self.L - fe.DOLFIN_EPS)

        def _left(x, on_boundary):
            return x[0] <= fe.DOLFIN_EPS

        self._right = _right
        self._left = _left

        # set the IC's
        if self.simulation == "dam_break":
            for init in [self.du, self.du_prev]:
                fe.assign(init.sub(1), ic)

            h_left = fe.Constant(5.0)
            h_right = fe.Constant(0.0)

            bc0 = fe.DirichletBC(W.sub(1), h_left, self._left)
            bc1 = fe.DirichletBC(W.sub(1), h_right, self._right)

            # add in boundary terms to the weak form
            self.F += v_h * u_prev * (
                self.H + h_right) * fe.ds - v_h * u_prev * (self.H + h_left) * fe.ds
        elif self.simulation == "tidal_flow":
            u_right = fe.Constant(0.0)

            bc0 = fe.DirichletBC(W.sub(1), self.tidal_bc(0), self._left)
            bc1 = fe.DirichletBC(W.sub(0), u_right, self._right)

        self.bcs = [bc0, bc1]

    @staticmethod
    def tidal_bc(t):
        return 4 - 4 * np.sin(np.pi * ((4 * t) / 86_400 + 0.5))

    def solve(self, t):
        if self.simulation == "tidal_flow":
            self.bcs[0] = fe.DirichletBC(self.W.sub(1), self.tidal_bc(t),
                                         self._left)

        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)
        fe.assign(self.du_prev, self.du)


class ShallowTwo:
    def __init__(self, control):
        # settings: L, nu, C
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.theta = control["theta"]

        self.mesh = fe.UnitSquareMesh(self.nx, self.nx)
        self.x = fe.SpatialCoordinate(self.mesh)
        self.x_coords = self.mesh.coordinates()
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        U = fe.FiniteElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        self.nu = 0.6
        self.C = 0.0025
        self.H = 20.


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simulation", default="dam_break")
    args = parser.parse_args()

    if args.simulation == "dam_break":
        nt = 240
        control = {
            "nx": 1000,
            "dt": 0.25,
            "theta": 1.0,
            "simulation": "dam_break"
        }

        u_lim = [0., 3.]
        h_lim = [0., 5.]
    elif args.simulation == "tidal_flow":
        nt = 3600
        control = {
            "nx": 1000,
            "dt": 2.5,
            "theta": 1.0,
            "simulation": "tidal_flow"
        }

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
    line, = ax.plot(swe.x_coords, 0*u_out[0])

    def animate(i):
        line.set_ydata(u_out[i])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, frames=nt, save_count=50)
    writer = animation.FFMpegWriter(
        fps=24, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f"figures/u_{args.simulation}.mp4", writer=writer)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_ylim(h_lim)
    line, = ax.plot(swe.x_coords, 0*h_out[0])

    def animate(i):
        line.set_ydata(h_out[i])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, frames=nt, save_count=50)
    writer = animation.FFMpegWriter(
        fps=24, metadata=dict(artist='Me'), bitrate=1800)
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
