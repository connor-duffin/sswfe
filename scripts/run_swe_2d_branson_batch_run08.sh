#!/bin/bash

RHO=5e-2
K_APPROX=50
K_FULL=100

python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.4 --rho $RHO --obs_skip 20 --tqdm_offset 0 \
	--k_approx $K_APPROX --k_full $K_FULL --compute_posterior
# & \1
# python3 scripts/run_swe_2d_branson_data.py \
# 	--dt 0.1 --rho $RHO --obs_skip 10 --tqdm_offset 1 \
# 	--k_approx $K_APPROX --k_full $K_FULL \
# 	--compute_posterior & \
# python3 scripts/run_swe_2d_branson_data.py \
# 	--dt 0.2 --rho $RHO --obs_skip 10 --tqdm_offset 2 \
# 	--k_approx $K_APPROX --k_full $K_FULL \
# 	--compute_posterior && fg
