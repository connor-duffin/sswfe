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

gmsh.initialize()

L = 5.46
H = 1.85
c_x = L / 2
c_y = H / 2
r = 0.05
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0

if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert(len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H/2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H/2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

# Create distance field from obstacle.
# Add threshold of mesh sizes based on the distance field
# LcMax -                  /--------
#                      /
# LcMin -o---------/
#        |         |       |
#       Point    DistMin DistMax
res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    # gmsh.fltk.run()


mesh, _, ft = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"


# Define boundary conditions
class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 0.02
        return values


# Navier-Stokes solver
# HACK(connor): basically a huge closure over all the mesh
class NSSplit:
    def __init__(self, dt, params):
        self.dt = Constant(mesh, PETSc.ScalarType(dt))
        self.mu = Constant(mesh, PETSc.ScalarType(params["mu"]))
        self.rho = Constant(mesh, PETSc.ScalarType(params["rho"]))

        # setup function spaces
        v_cg2 = VectorElement("CG", mesh.ufl_cell(), 2)
        s_cg1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        self.V = FunctionSpace(mesh, v_cg2)
        self.Q = FunctionSpace(mesh, s_cg1)
        self.fdim = mesh.topology.dim - 1

    def setup_form(self):
        # Inlet
        self.u_inlet = Function(self.V)
        self.inlet_velocity = InletVelocity(t)
        self.u_inlet.interpolate(self.inlet_velocity)
        bcu_inflow = dirichletbc(self.u_inlet,
                                 locate_dofs_topological(
                                     self.V, self.fdim, ft.find(inlet_marker)))

        # Walls
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_walls = dirichletbc(u_nonslip,
                                locate_dofs_topological(
                                    self.V, self.fdim, ft.find(wall_marker)), self.V)

        # Obstacle
        bcu_obstacle = dirichletbc(u_nonslip,
                                   locate_dofs_topological(
                                       self.V, self.fdim, ft.find(obstacle_marker)), self.V)
        self.bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        # Outlet
        bcp_outlet = dirichletbc(PETSc.ScalarType(0.),
                                 locate_dofs_topological(self.Q, self.fdim, ft.find(outlet_marker)), self.Q)
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

        f = Constant(mesh, PETSc.ScalarType((0, 0)))
        F1 = self.rho / self.dt * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1,
                        0.5 * nabla_grad(u + u_n)), v) * dx
        F1 += 0.5 * self.mu * inner(grad(u + u_n),
                                    grad(v))*dx - dot(p_, div(v)) * dx
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
        self.solver1 = PETSc.KSP().create(mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # solver for step 2
        self.solver2 = PETSc.KSP().create(mesh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # solver for step 3
        self.solver3 = PETSc.KSP().create(mesh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solve(self, t):
        self.inlet_velocity.t = t
        self.u_inlet.interpolate(self.inlet_velocity)

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
        with self.u_.vector.localForm() as loc_, self.u_n.vector.localForm() as loc_n, self.u_n1.vector.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)


t = 0
T = 120
dt = 1e-4
num_steps = int(T / dt)

params = dict(mu=0.1, rho=1000)
ns = NSSplit(dt, params)
ns.setup_form()

u_out = XDMFFile(mesh.comm, "outputs/dfg2D-3-u.bp", "w")
p_out = XDMFFile(mesh.comm, "outputs/dfg2D-3-p.bp", "w")
for out in [u_out, p_out]:
    out.write_mesh(mesh)

u_out.write_function(ns.u_, t)
p_out.write_function(ns.p_, t)

progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)

    # Update current time step
    t += dt

    # solve and write solutions to file
    ns.solve(t)
    u_out.write_function(ns.u_, t)
    p_out.write_function(ns.p_, t)

u_out.close()
p_out.close()
