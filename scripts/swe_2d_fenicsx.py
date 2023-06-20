import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from ns_2d_fenicsx import NSSplit
from tqdm import tqdm

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

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import ufl

from mpi4py import MPI
from petsc4py import PETSc

import faulthandler

d = 0.04  # 4cm
L = 25 * d
H = 14 * d
c_x = 5 * d
c_y = 7 * d
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0

gmsh.initialize()

if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addRectangle(4.5 * d, 6.5 * d, 0, d, d)

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
res_min = d / 10
res_max = 0.1 * H
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", res_max)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", d)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    # gmsh.fltk.run()


msh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"
fdim = msh.topology.dim - 1
# facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


# Define boundary conditions
class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]),
                          dtype=PETSc.ScalarType)
        values[0] = 0.535
        return values


t = 0.
T = 10.
dt = 1e-3
nt = int(T/dt)

nu = Constant(msh, PETSc.ScalarType(1e-3))
g = Constant(msh, PETSc.ScalarType(9.8))
H = Constant(msh, PETSc.ScalarType(0.16))

# setup mixed function space
v_cg2 = ufl.VectorElement("CG", msh.ufl_cell(), 2)
s_cg1 = ufl.FiniteElement("CG", msh.ufl_cell(), 1)
W = FunctionSpace(msh, v_cg2 * s_cg1)

n = ufl.FacetNormal(msh)
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

v, q = ufl.TestFunctions(W)
du = Function(W)
du_prev = Function(W)

# set current functions
u, h = ufl.split(du)
u_prev, h_prev = ufl.split(du_prev)

# inflow BC
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
u_inlet = Function(V)
u_inlet.interpolate(InletVelocity(0.))
u_nonslip = Function(V)
u_nonslip.x.array[:] = 0.

theta = 1.
u_mid = theta * u + (1 - theta) * u_prev
h_mid = theta * h + (1 - theta) * h_prev
F = (ufl.inner(u - u_prev, v) / dt * ufl.dx  # mass term u
     + ufl.inner(ufl.dot(u_mid, ufl.nabla_grad(u_mid)), v) * ufl.dx  # advection
     + nu * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx  # dissipation
     + g * ufl.inner(ufl.grad(h_mid), v) * ufl.dx  # surface term
     + ufl.inner(h - h_prev, q) / dt * ufl.dx  # mass term h
     - ufl.inner((H + h) * u, ufl.grad(q)) * ufl.dx  # continuity term
     + (H + h) * q * ufl.inner(u_inlet, n) * ds(2)  # inlet BC
     + (H + h) * q * ufl.inner(u_inlet, n) * ds(3)  # outlet flather BC
     + (H + h) * q * ufl.sqrt(g / H) * h_mid * ds(3))  # outlet flather BC

bcu_inflow = dirichletbc(
    u_inlet, locate_dofs_topological(W.sub(0), fdim, ft.find(inlet_marker)))

bcu_obstacle = dirichletbc(
    u_nonslip, locate_dofs_topological(W.sub(0), fdim, ft.find(obstacle_marker)))

bcs = [bcu_inflow, bcu_obstacle]
problem = NonlinearProblem(F, du, bcs=bcs)
solver = NewtonSolver(mesh_comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-3
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

for i in tqdm(range(nt)):
    r = solver.solve(du)
    du_prev.x.array[:] = du.x.array
