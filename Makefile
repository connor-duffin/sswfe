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


# 1d tidal flow
# -------------
tidal_output_dir = outputs/swe-tidal
nus = 100.0 1e-1 5e-2 4e-2 3e-2 2.5e-2 2e-2 1e-2
tidal_1d_output_files = $(foreach nu,$(nus),$(tidal_output_dir)/1d-nu-$(nu).h5)

$(tidal_output_dir)/1d-nu-%.h5: scripts/run_swe_1d_tidal.py
	python3 $< \
		--nu $* --output_file $@ \
		--nx 400 --dt 4.0 --n_cycles 50 --nt_save 100

all_tidal_1d_outputs: $(tidal_1d_output_files)
	@echo $(tidal_1d_output_files)

clean_all_tidal_1d_outputs: $(tidal_1d_output_files)
	rm -f $(tidal_1d_output_files)


# 1d immersed bump
# ----------------
bump_output_dir = outputs/swe-bump
nus = 1 1e-1 1e-2 1e-3 1e-4
bump_linear = $(bump_output_dir)/1d-linear.h5
bump_nonlinear = $(foreach nu,$(nus),$(bump_output_dir)/1d-nu-$(nu).h5)
bump_1d_output_files = $(bump_linear) $(bump_nonlinear)

$(bump_output_dir)/1d-nu-%.h5: scripts/run_swe_1d_bump.py
	python3 $< \
		--nu $* --output_file $@ \
		--nx 500 --dt 0.01 --nt_save 10

$(bump_output_dir)/1d-linear.h5: scripts/run_swe_1d_bump.py
	python3 $< \
		--output_file $@ --linear \
		--nx 500 --dt 0.01 --nt_save 10

all_bump_1d_outputs: $(bump_1d_output_files)
	@echo $(bump_1d_output_files)

clean_all_bump_1d_outputs: $(bump_1d_output_files)
	rm -f $(bump_1d_output_files)


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
