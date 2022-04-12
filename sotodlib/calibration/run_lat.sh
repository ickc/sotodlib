#!/bin/bash

ntasks=8
indir=/home/zhileixu/data/simulated_tod/LAT/i6
outdir=/home/zhileixu/data/fitting_results

mpirun -n ${ntasks} ./calibration_tod_analysis.py \
        --indir ${indir} \
        --outdir ${outdir} \
        --f_format g3 \
        --tele LAT \
        --tube i6 \
        --band f090 \
        --wafer w13 \
        --sso_name Saturn \
        --year 2022 \
        --month 9 \
        --day 1 \
        --highpass \
        --cutoff 0.2 \
        