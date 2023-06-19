import h5py
import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import create_adjacencylist
from dolfinx.geometry import (BoundingBoxTree, compute_collisions,
                              compute_colliding_cells)
from dolfinx.io import (XDMFFile, VTKFile, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities

from ufl import (FacetNormal, FiniteElement, Identity, Measure,
                 TestFunction, TrialFunction, VectorElement, as_vector, div,
                 dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)


gdim = 2

class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 0.01
        return values

# Navier-Stokes solver
class NSSplit:
    def __init__(self, mesh, dt, params):
        # L = 5.46
        # H = 1.85
        # c_x = L / 2
        # c_y = H / 2
        # r = 0.05
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        if type(mesh) == str:
            self.msh, _, self.ft = gmshio.read_from_msh(
                mesh, mesh_comm, model_rank, gdim=gdim)
            self.ft.name = "Facet markers"
        elif type(mesh) == tuple:
            self.msh, self.ft = mesh
        else:
            print("Expected either tuple (msh, ft) or string for mesh")
            raise ValueError

        self.dt = Constant(self.msh, PETSc.ScalarType(dt))
        self.mu = Constant(self.msh, PETSc.ScalarType(params["mu"]))
        self.rho = Constant(self.msh, PETSc.ScalarType(params["rho"]))

        # setup function spaces
        v_cg2 = VectorElement("CG", self.msh.ufl_cell(), 2)
        s_cg1 = FiniteElement("CG", self.msh.ufl_cell(), 1)

        self.V = FunctionSpace(self.msh, v_cg2)
        self.V_dof_coordinates = self.V.tabulate_dof_coordinates()

        self.Q = FunctionSpace(self.msh, s_cg1)
        self.Q_dof_coordinates = self.Q.tabulate_dof_coordinates()
        self.fdim = self.msh.topology.dim - 1

        # self.u_inlet = Function(self.V)
        # self.inlet_velocity = InletVelocity(t)
        # self.u_inlet.interpolate(self.inlet_velocity)
        # or just a const.
        # self.u_inlet = np.array((0.01, 0), dtype=PETSc.ScalarType)

    def setup_form(self):
        # HACK(connor): hardcode and mark relevant parts of the domain
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

        # Walls
        u_nonslip = np.array((0, 0), dtype=PETSc.ScalarType)
        bcu_walls = dirichletbc(
            u_nonslip,
            locate_dofs_topological(self.V, self.fdim, self.ft.find(wall_marker)),
            self.V)

        # Obstacle
        bcu_obstacle = dirichletbc(
            u_nonslip,
            locate_dofs_topological(self.V, self.fdim, self.ft.find(obstacle_marker)),
            self.V)

        # Inlet
        if type(self.u_inlet) == np.ndarray:
            bcu_inflow = dirichletbc(
                self.u_inlet,
                locate_dofs_topological(self.V, self.fdim, self.ft.find(inlet_marker)),
                self.V)
        else:
            bcu_inflow = dirichletbc(
                self.u_inlet,
                locate_dofs_topological(self.V, self.fdim, self.ft.find(inlet_marker)))
        self.bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        # Outlet
        bcp_outlet = dirichletbc(
            PETSc.ScalarType(0.),
            locate_dofs_topological(self.Q, self.fdim, self.ft.find(outlet_marker)),
            self.Q)
        self.bcp = [bcp_outlet]

        # trial/test functions used for form creation, only.
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)

        self.u_ = Function(self.V)
        self.u_.name = "u"
        self.u_s = u_s = Function(self.V)
        self.u_n = u_n = Function(self.V)
        self.u_n1 = u_n1 = Function(self.V)

        self.p_ = p_ = Function(self.Q)
        self.p_.name = "p"
        self.phi = phi = Function(self.Q)

        f = Constant(self.msh, PETSc.ScalarType((0, 0)))
        F1 = self.rho / self.dt * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1,
                        0.5 * nabla_grad(u + u_n)), v) * dx
        F1 += 0.5 * self.mu * inner(grad(u + u_n), grad(v))*dx - dot(p_, div(v)) * dx
        F1 += dot(f, v) * dx

        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        self.A1 = create_matrix(self.a1)
        self.b1 = create_vector(self.L1)

        self.a2 = form(dot(grad(p), grad(q))*dx)
        self.L2 = form(-self.rho / self.dt * dot(div(self.u_s), q) * dx)
        self.A2 = assemble_matrix(self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.b2 = create_vector(self.L2)

        self.a3 = form(self.rho * dot(u, v)*dx)
        self.L3 = form(self.rho * dot(u_s, v)*dx
                       - self.dt * dot(nabla_grad(phi), v)*dx)
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(self.L3)

        # solver for step 1
        self.solver1 = PETSc.KSP().create(self.msh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # solver for step 2
        self.solver2 = PETSc.KSP().create(self.msh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # solver for step 3
        self.solver3 = PETSc.KSP().create(self.msh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solve(self, t):
        # step 1: tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.vector)
        self.u_s.x.scatter_forward()

        # step 2: pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.phi.vector)
        self.phi.x.scatter_forward()

        self.p_.vector.axpy(1, self.phi.vector)
        self.p_.x.scatter_forward()

        # step 3: velocity correction step
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.vector)
        self.u_.x.scatter_forward()

        # update variable with solution form this time step
        with (self.u_.vector.localForm() as loc_,
              self.u_n.vector.localForm() as loc_n,
              self.u_n1.vector.localForm() as loc_n1):
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)
