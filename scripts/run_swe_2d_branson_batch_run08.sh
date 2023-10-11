#!/bin/bash

RHO=1
K_APPROX=128
K_FULL=256
DT=0.05

python3 scripts/run_swe_2d_branson_data.py \
	--dt $DT --rho $RHO --obs_skip 1 --tqdm_offset 0 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt $DT --rho $RHO --obs_skip 5 --tqdm_offset 1 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt $DT --rho $RHO --obs_skip 10 --tqdm_offset 2 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior & \
python3 scripts/run_swe_2d_branson_data.py \
	--dt $DT --rho $RHO --obs_skip 20 --tqdm_offset 3 \
	--k_approx $K_APPROX --k_full $K_FULL \
	--compute_posterior && fg
