import gmsh
import sys

cm = 1e-2
domain_width = 1.85
domain_length = 6.
cylinder_diameter = 10 * cm

gmsh.initialize()
gmsh.model.add("domain-with-cylinder")
h = 5 * cm

factory = gmsh.model.occ
factory.addPoint(0., 0., 0, h, 1)
factory.addPoint(0., domain_width, 0, h, 2)
factory.addPoint(domain_length, domain_width, 0, h, 3)
factory.addPoint(domain_length, 0., 0, h, 4)

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

factory.addLine(1, 2, 1)  # inflow
factory.addLine(2, 3, 2)  # top
factory.addLine(3, 4, 3)  # outflow
factory.addLine(4, 1, 4)  # bottom

factory.addCircleArc(6, 5, 8, 5)  # left to bot
factory.addCircleArc(7, 5, 8, 6)  # right to bot
factory.addCircleArc(6, 5, 9, 7)  # left to top
factory.addCircleArc(7, 5, 9, 8)  # right to top

factory.addCurveLoop([5, -6, 8, -7], 9)
factory.addCurveLoop([1, 2, 3, 4], 10)

# generate the surface and synchronize
factory.addPlaneSurface([9, 10], 1)
factory.synchronize()

gmsh.model.addPhysicalGroup(1, [1], name="Inflow")
gmsh.model.addPhysicalGroup(1, [2], name="Top")
gmsh.model.addPhysicalGroup(1, [3], name="Outflow")
gmsh.model.addPhysicalGroup(1, [4], name="Bottom")
gmsh.model.addPhysicalGroup(1, [9, 10], name="Cylinder")
gmsh.model.addPhysicalGroup(2, [1], name="Surface")

gmsh.model.mesh.generate(2)

gmsh.write("data/cylinder.msh")

if '-popup' in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
