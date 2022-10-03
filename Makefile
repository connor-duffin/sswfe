# syncing
# -------
rsb:
		rsync -avzz \
		cambox:/home/cpd32@ad.eng.cam.ac.uk/projects/20220609-swfe/figures/ \
		/Users/connor/Projects/statfluids/20220609-swfe/figures/

rsd:
		rsync -avzz \
		cambox:/home/cpd32@ad.eng.cam.ac.uk/projects/20220609-swfe/docs/ \
		/Users/connor/Projects/statfluids/20220609-swfe/docs/


# meshes
# ------
mesh/channel.msh:
		python3 scripts/generate_mesh_hole.py --popup $@

mesh/channel.xdmf: mesh/channel.msh
		python3 scripts/convert_mesh.py $< $@

mesh/channel-hole.msh:
		python3 scripts/generate_mesh_hole.py --add_cylinder --popup $@

mesh/channel-hole.xdmf: mesh/channel-hole.msh
		python3 scripts/convert_mesh.py $< $@

mesh/channel-piggott.msh:
		python3 scripts/generate_mesh_square_hole.py --add_hole $@

mesh/channel-piggott.xdmf: mesh/channel-piggott.msh
		python3 scripts/convert_mesh.py $< $@


# laminar flow
# ------------
outputs/swe-laminar.h5: mesh/channel.xdmf
		python3 scripts/run_swe_2d_channel.py $< $@

outputs/swe-laminar-ibp.h5: mesh/channel.xdmf
		python3 scripts/run_swe_2d_channel.py --integrate_continuity_by_parts $< $@

laminar_examples: outputs/swe-laminar.h5 outputs/swe-laminar-ibp.h5

laminar_figures: mesh/channel.xdmf outputs/swe-laminar.h5
		python3 scripts/plot_channel.py \
		mesh/channel.xdmf \
		outputs/swe-laminar.h5 \
		figures/swe-laminar/

laminar_ibp_figures: mesh/channel.xdmf outputs/swe-laminar-ibp.h5
		python3 scripts/plot_channel.py \
				--integrate_continuity_by_parts \
				mesh/channel.xdmf \
				outputs/swe-laminar-ibp.h5 \
				figures/swe-laminar-ibp/


# cylinder examples
# -----------------
outputs/swe-cylinder.h5: mesh/channel-hole.xdmf
		python3 scripts/run_swe_2d_channel.py \
				--cylinder mesh/channel-hole.xdmf $@

outputs/swe-cylinder-refined.h5: mesh/channel-hole-refined.xdmf
		python3 scripts/run_swe_2d_channel.py \
				--cylinder mesh/channel-hole-refined.xdmf $@

outputs/swe-cylinder-ibp.h5: mesh/channel-hole.xdmf
		python3 scripts/run_swe_2d_channel.py \
				--cylinder --integrate_continuity_by_parts mesh/channel-hole.xdmf $@

cylinder_figures: mesh/channel-hole.xdmf outputs/swe-cylinder.h5
		python3 scripts/plot_channel.py \
		mesh/channel-hole.xdmf \
		outputs/swe-cylinder.h5 \
		figures/swe-cylinder/
