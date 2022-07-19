import gmsh
import sys

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("output_file", type=str)
parser.add_argument("--add_cylinder", action="store_true")
parser.add_argument("--popup", action="store_true")
args = parser.parse_args()

cm = 1e-2
domain_width = 1.85
domain_length = 6.
cylinder_diameter = 10 * cm

# pad out the domain to ensure outflow condition
if args.add_cylinder:
    domain_length += 0.2

gmsh.initialize()
gmsh.model.add("channel")
h = 5 * cm

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
    factory.addPoint(domain_width / 2, domain_width / 2, 0, h / 5, 5)
    factory.addPoint(domain_width / 2 - cylinder_diameter, domain_width / 2, 0,
                    h / 5, 6)
    factory.addPoint(domain_width / 2 + cylinder_diameter, domain_width / 2, 0,
                    h / 5, 7)
    factory.addPoint(domain_width / 2, domain_width / 2 - cylinder_diameter, 0,
                    h / 5, 8)
    factory.addPoint(domain_width / 2, domain_width / 2 + cylinder_diameter, 0,
                    h / 5, 9)

    factory.addCircleArc(6, 5, 8, 5)  # left to bot
    factory.addCircleArc(7, 5, 8, 6)  # right to bot
    factory.addCircleArc(6, 5, 9, 7)  # left to top
    factory.addCircleArc(7, 5, 9, 8)  # right to top

    factory.addCurveLoop([5, -6, 8, -7], 9)

# generate the surface and synchronize
if args.add_cylinder:
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

gmsh.model.mesh.generate(2)

gmsh.write(args.output_file)

if args.popup:
    gmsh.fltk.run()

gmsh.finalize()
