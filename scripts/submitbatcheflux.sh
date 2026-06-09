#!/bin/bash

for YEAR in {1979..2023}; do
    qsub -v YEAR=$YEAR submit_years_efluxes.pbs
done
