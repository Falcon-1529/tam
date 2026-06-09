#!/bin/bash

for YEAR in {1979..2023}; do
    qsub -v YEAR=$YEAR submit_yearsprecip.pbs
done