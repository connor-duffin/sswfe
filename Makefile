data/channel.msh:
		python3 scripts/generate_mesh_hole.py $@

data/channel.xdmf: data/channel.msh
		python3 scripts/convert_mesh.py $< $@

data/channel-hole.msh:
		python3 scripts/generate_mesh_hole.py --add_cylinder $@

data/channel-hole.xdmf: data/channel-hole.msh
		python3 scripts/convert_mesh.py $< $@
