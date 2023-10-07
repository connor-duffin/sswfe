#!/bin/bash

RHO=0.1
K_APPROX=128
K_FULL=256

python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.05 --rho $RHO --tqdm_offset 0 \
	--k_approx $K_APPROX --k_full $K_FULL & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.05 --rho $RHO --tqdm_offset 1 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.01 --rho $RHO --tqdm_offset 2 \
	--k_approx $K_APPROX --k_full $K_FULL & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.01 --rho $RHO --tqdm_offset 3 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.02 --rho $RHO --tqdm_offset 4 \
	--k_approx $K_APPROX --k_full $K_FULL & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt 0.02 --rho $RHO --tqdm_offset 5 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior && fg
