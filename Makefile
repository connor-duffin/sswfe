# meshes
# ------
mesh/branson-mesh-nondim.msh:
	python3 scripts/generate_mesh_hole.py --add_cylinder $@

%.xdmf: %.msh
	python3 scripts/convert_mesh.py $< $@


# computations: Run08
# -------------------
outputs/branson-run08-swe-prior-testing.h5:
	python3 scripts/run_swe_2d_branson_data.py $@ \
		--log_file log/branson-run08-swe-prior-testing.log

outputs/branson-run08-swe-posterior-testing.h5:
	python3 scripts/run_swe_2d_branson_data.py $@ \
		--compute_posterior \
		--log_file log/branson-run08-swe-posterior-testing.log
