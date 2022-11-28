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
ks = 4 8 16 64 128
cs = 7 11 12 15 20
nus = 1e-6 1e-4 1e-2 1 100
nt_skips = 1 5 10 25 100
bump_output_dir = outputs/swe-bump
n_threads = 4

# priors
bump_priors_linear:  # data/h_bump.nc
	python3 scripts/run_filter_swe_1d_bump.py \
		--linear --n_threads $(n_threads) --nu 0. --nt_skip 1 --k 32 \
		--data_file data/h_bump.nc --output_dir $(bump_output_dir)

bump_priors_nonlinear:  # data/h_bump.nc
	python3 scripts/run_filter_swe_1d_bump.py \
		--n_threads $(n_threads) --nu $(nus) --nt_skip 1 --k 32 \
		--data_file data/h_bump.nc --output_dir $(bump_output_dir)

# posteriors
bump_filters_nonlinear:  # data/h_bump.nc
	python3 scripts/run_filter_swe_1d_bump.py \
		--n_threads $(n_threads) --nt_skip 20 --k 32 --nu $(nus) --c $(cs) --posterior \
		--data_file data/h_bump.nc --output_dir $(bump_output_dir)

# bump_filters_nonlinear:  # data/h_bump.nc
# 	python3 scripts/run_filter_swe_1d_bump.py \
# 		--n_threads $(n_threads) --nt_skip $(nt_skips) --k 32 --nu $(nus) --posterior \
# 		--data_file data/h_bump.nc --output_dir $(bump_output_dir)

bump_filters_linear:  # data/h_bump.nc
	python3 scripts/run_filter_swe_1d_bump.py \
		--linear --n_threads $(n_threads) --nu 0. --nt_skip $(nt_skips) --k 32 --posterior \
		--data_file data/h_bump.nc --output_dir $(bump_output_dir)

# deterministic models
$(bump_output_dir)/nu-%.h5: scripts/run_swe_1d_bump.py
	python3 $< \
		--nu $* --output_file $@ \
		--nx 500 --dt 0.01 --nt_save 10

$(bump_output_dir)/linear.h5: scripts/run_swe_1d_bump.py
	python3 $< \
		--output_file $@ --linear \
		--nx 500 --dt 0.01 --nt_save 10

all_bump_prior: bump_priors_nonlinear bump_priors_linear

all_bump_post: bump_filters_nonlinear bump_filters_linear

all_bump_prior_post: all_bump_prior all_bump_post

clean_all_bump_outputs:
	rm $(bump_output_dir)/*


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
