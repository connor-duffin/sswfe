import gmsh
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--add_hole", action="store_true")
parser.add_argument("--popup", action="store_true")
parser.add_argument("output_file", type=str)
args = parser.parse_args()

d = 0.04
domain_width = 14 * d
domain_length = 25 * d
h = d / 4  # 1cm resolution

gmsh.initialize()
gmsh.model.add("channel")

factory = gmsh.model.occ
factory.addPoint(0., 0., 0, h, 1)
factory.addPoint(0., domain_width, 0, h, 2)
factory.addPoint(domain_length, domain_width, 0, h, 3)
factory.addPoint(domain_length, 0., 0, h, 4)

factory.addLine(1, 2, 1)  # inflow
factory.addLine(2, 3, 2)  # top
factory.addLine(3, 4, 3)  # outflow
factory.addLine(4, 1, 4)  # bot
factory.addCurveLoop([1, 2, 3, 4], 10)  # 2d surface

if args.add_hole:
    factory.addPoint(4.5 * d, 6.5 * d, 0, h, 5)  # left bot
    factory.addPoint(4.5 * d, 7.5 * d, 0, h, 6)  # left top
    factory.addPoint(5.5 * d, 7.5 * d, 0, h, 7)  # right top
    factory.addPoint(5.5 * d, 6.5 * d, 0, h, 8)  # right bot

    factory.addLine(5, 6, 5)  # left
    factory.addLine(6, 7, 6)  # top
    factory.addLine(7, 8, 7)  # right
    factory.addLine(8, 5, 8)  # bot
    factory.addCurveLoop([5, 6, 7, 8], 9)  # cylinder
    factory.addPlaneSurface([9, 10], 1)
else:
    factory.addPlaneSurface([10], 1)

factory.synchronize()

gmsh.model.addPhysicalGroup(1, [1], name="Inflow")
gmsh.model.addPhysicalGroup(1, [2], name="Top")
gmsh.model.addPhysicalGroup(1, [3], name="Outflow")
gmsh.model.addPhysicalGroup(1, [4], name="Bottom")
gmsh.model.addPhysicalGroup(2, [1], name="Surface")

gmsh.model.mesh.generate(2)
gmsh.write(args.output_file)

if args.popup:
    gmsh.fltk.run()

gmsh.finalize()
