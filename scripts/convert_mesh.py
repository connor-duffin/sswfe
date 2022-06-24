import meshio
import fenics as fe

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
args = parser.parse_args()

# read in mesh
msh = meshio.read(args.input_file)

# extract relevant information
cell_type = "triangle"
cells = msh.get_cells_type(cell_type)
points = msh.points[:, :2]
triangle_msh = meshio.Mesh(points=points, cells={cell_type: cells})

# output mesh
meshio.write(args.output_file, triangle_msh)
