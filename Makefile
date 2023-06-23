# meshes
# ------
mesh/branson.msh:
	python3 scripts/generate_mesh_hole.py --add_cylinder $@

%.xdmf: %.msh
	python3 scripts/convert_mesh.py $< $@
