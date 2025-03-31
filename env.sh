#!/bin/bash

# Define some basic environmental variables before launching the suite

# Load the analysis3 conda environment
module use /g/data/hh5/public/modules
module load conda/analysis3

# Root directory for this repo
export ROOT=/home/548/${USER}/aus-historical-solar-droughts
export MODULES=${ROOT}/modules
export CONFIG=${ROOT}/config

# Append to our python path
export PYTHONPATH=${MODULES}:${CONFIG}:${PYTHONPATH}