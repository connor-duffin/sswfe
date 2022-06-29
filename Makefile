rsb:
		rsync -avzz \
		cambox:/home/cpd32@ad.eng.cam.ac.uk/projects/20220609-swfe/figures/ \
		/Users/connor/Projects/statfluids/20220609-swfe/figures/

mesh/channel.msh:
		python3 scripts/generate_mesh_hole.py $@

mesh/channel.xdmf: mesh/channel.msh
		python3 scripts/convert_mesh.py $< $@

mesh/channel-hole.msh:
		python3 scripts/generate_mesh_hole.py --add_cylinder $@

mesh/channel-hole.xdmf: mesh/channel-hole.msh
		python3 scripts/convert_mesh.py $< $@

outputs/swe-laminar.h5: mesh/channel.xdmf
		python3 scripts/run_swe_2d_laminar.py $< $@

outputs/swe-laminar-ibp.h5: mesh/channel.xdmf
		python3 scripts/run_swe_2d_laminar.py --integrate_continuity_by_parts $< $@

laminar_examples: outputs/swe-laminar.h5 outputs/swe-laminar-ibp.h5

laminar_figures: mesh/channel.xdmf outputs/swe-laminar.h5
		python3 scripts/plot_laminar.py \
		mesh/channel.xdmf \
		outputs/swe-laminar.h5 \
		figures/swe-laminar/

laminar_ibp_figures: mesh/channel.xdmf outputs/swe-laminar-ibp.h5
		python3 scripts/plot_laminar.py \
		--integrate_continuity_by_parts \
		mesh/channel.xdmf \
		outputs/swe-laminar-ibp.h5 \
		figures/swe-laminar-ibp/

