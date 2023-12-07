import gmsh
import meshio
import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("output_file", type=str, help='Output .xdmf file to store the mesh in')
parser.add_argument("--add_cylinder", action="store_true", help='Flag to add the cylinder into the mesh')
parser.add_argument("--refinement_factor", type=float, default=1., help='Mesh size scale factor (large => coarser)')
parser.add_argument("--popup", action="store_true", help='Flag to draw Tk GUI')
args = parser.parse_args()

temp_file, ext = os.path.splitext(args.output_file)
temp_file += ".msh"

# cm = 1e-2
# domain_width = 0.41
# domain_length = 2.2
domain_width = 5.
domain_length = 20.

# cylinder setup
cyl_diameter = 1.
cyl_radius = cyl_diameter / 2
cyl_center = [domain_length / 2, domain_width / 2]

gmsh.initialize()
gmsh.model.add("channel")
h = args.refinement_factor * cyl_diameter / 10

factory = gmsh.model.occ
factory.addPoint(0., 0., 0, h, 1)
factory.addPoint(0., domain_width, 0, h, 2)
factory.addPoint(domain_length, domain_width, 0, h, 3)
factory.addPoint(domain_length, 0., 0, h, 4)

factory.addLine(1, 2, 1)  # inflow
factory.addLine(2, 3, 2)  # top
factory.addLine(3, 4, 3)  # outflow
factory.addLine(4, 1, 4)  # bottom

factory.addCurveLoop([1, 2, 3, 4], 10)

if args.add_cylinder:
    # more refined about the cylinder
    factory.addPoint(cyl_center[0], cyl_center[1], 0, h / 5, 5)
    factory.addPoint(cyl_center[0] - cyl_radius, cyl_center[1], 0, h / 5, 6)
    factory.addPoint(cyl_center[0] + cyl_radius, cyl_center[1], 0, h / 5, 7)
    factory.addPoint(cyl_center[0], cyl_center[1] - cyl_radius, 0, h / 5, 8)
    factory.addPoint(cyl_center[0], cyl_center[1] + cyl_radius, 0, h / 5, 9)

    factory.addCircleArc(6, 5, 8, 5)  # left to bot
    factory.addCircleArc(7, 5, 8, 6)  # right to bot
    factory.addCircleArc(6, 5, 9, 7)  # left to top
    factory.addCircleArc(7, 5, 9, 8)  # right to top

    factory.addCurveLoop([5, -6, 8, -7], 9)
    factory.addPlaneSurface([9, 10], 1)
else:
    factory.addPlaneSurface([10], 1)

factory.synchronize()

gmsh.model.addPhysicalGroup(1, [1], name="Inflow")
gmsh.model.addPhysicalGroup(1, [2], name="Top")
gmsh.model.addPhysicalGroup(1, [3], name="Outflow")
gmsh.model.addPhysicalGroup(1, [4], name="Bottom")
gmsh.model.addPhysicalGroup(2, [1], name="Surface")

if args.add_cylinder:
    gmsh.model.addPhysicalGroup(1, [9, 10], name="Cylinder")

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

if args.popup:
    gmsh.fltk.run()

gmsh.write(temp_file)
gmsh.finalize()

print(f'Formatting temp. mesh ({temp_file}) to FEniCS ({args.output_file})')
msh = meshio.read(temp_file)

# extract relevant information
cell_type = "triangle"
cells = msh.get_cells_type(cell_type)
points = msh.points[:, :2]
triangle_msh = meshio.Mesh(points=points, cells={cell_type: cells})

# output mesh
meshio.write(args.output_file, triangle_msh)
