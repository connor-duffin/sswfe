import meshio
import fenics as fe

from argparse import ArgumentParser


def main(input_file, output_file):
    # read in mesh
    msh = meshio.read(input_file)

    # extract relevant information
    cell_type = "triangle"
    cells = msh.get_cells_type(cell_type)
    points = msh.points[:, :2]
    triangle_msh = meshio.Mesh(points=points, cells={cell_type: cells})

    # output mesh
    meshio.write(output_file, triangle_msh)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    main(args.input_file, args.output_file)

    # main("data/cylinder.msh", "data/cylinder.xdmf")
    # mesh = fe.Mesh()
    # f = fe.XDMFFile("data/cylinder.xdmf")
    # f.read(mesh)
    # fe.plot(mesh)

    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.close()
