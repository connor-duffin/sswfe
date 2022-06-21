import gmsh
from argparse import ArgumentParser


def main(dx, output_file, popup):
    gmsh.initialize()
    gmsh.model.add("unit-square")

    gmsh.model.geo.addPoint(0, 0, 0, dx, 1)
    gmsh.model.geo.addPoint(0, 1, 0, dx, 2)
    gmsh.model.geo.addPoint(1, 1, 0, dx, 3)
    gmsh.model.geo.addPoint(1, 0, 0, dx, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
    gmsh.model.addPhysicalGroup(2, [1], name="Unit Square")

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(output_file)

    if popup:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dx", default=0.1, type=float)  # mesh-refinement
    parser.add_argument("--output_file", type=str)  # store output
    parser.add_argument("-popup", action="store_true")  # defaults to false
    args = parser.parse_args()

    main(args.dx, args.output_file, args.popup)
