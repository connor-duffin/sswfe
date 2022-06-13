""" Solve the Shallow-water equations in non-conservative form. """
import fenics as fe
import matplotlib.pyplot as plt


class PiecewiseIC(fe.UserExpression):
    def eval(self, values, x):
        if x[0] < 1000. + fe.DOLFIN_EPS:
            values[0] = 5.
        else:
            values[0] = 0.


nx = 400
mesh = fe.IntervalMesh(nx, 0, 2000)

boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

U = fe.FiniteElement("P", mesh.ufl_cell(), 2)
H = fe.FiniteElement("P", mesh.ufl_cell(), 1)
TH = fe.MixedElement([U, H])
W = fe.FunctionSpace(mesh, TH)

p = PiecewiseIC()
# HACK: overwrite U and H with new FunctionSpaces
V = fe.FunctionSpace(mesh, "CG", 1)
ic = fe.interpolate(p, V)

du = fe.Function(W)
u, h = fe.split(du)

du_prev = fe.Function(W)
u_prev, h_prev = fe.split(du_prev)

v_u, v_h = fe.TestFunctions(W)

nu = fe.Constant(1.0)
g = fe.Constant(9.8)
C = fe.Constant(0.0)
H = fe.Constant(5.0)
dt = fe.Constant(0.025)

theta = 1.0
u_mid = theta * u + (1 - theta) * u_prev
h_mid = theta * h + (1 - theta) * h_prev
u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

h_left = fe.Constant(5.0)
h_right = fe.Constant(0.0)

F = (fe.inner(u - u_prev, v_u) / dt * fe.dx
     + fe.inner(h - h_prev, v_h) / dt * fe.dx
     + u_prev * u_mid.dx(0) * v_u * fe.dx
     + nu * 2 * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx
     + g * h_mid.dx(0) * v_u * fe.dx
     + C * u_mag * u_mid * v_u / (H + h_prev) * fe.dx
     - ((H + h_mid) * u_mid * v_h.dx(0)) * fe.dx
     + v_h * u_prev * (H + h_right) * fe.ds - v_h * u_prev * (H + h_left) * fe.ds)

# set the IC's
for init in [du, du_prev]:
    fe.assign(init.sub(1), ic)


# set the BC's
def right(x, on_boundary):
    return x[0] >= (2000 - fe.DOLFIN_EPS)


def left(x, on_boundary):
    return x[0] <= fe.DOLFIN_EPS


bc0 = fe.DirichletBC(W.sub(1), h_left, left)
bc1 = fe.DirichletBC(W.sub(1), h_right, right)

bcs = [bc0, bc1]

J = fe.derivative(F, du)

for i in range(2400):
    fe.solve(F == 0, du, bcs=bcs, J=J)
    fe.assign(du_prev, du)

fe.plot(u)
plt.show()
plt.close()

fe.plot(h)
plt.show()
plt.close()
