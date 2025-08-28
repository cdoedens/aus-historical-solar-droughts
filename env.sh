#!/bin/bash

# Define some basic environmental variables before launching the suite

# Load the analysis3 conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

# Root directory for this repo
export ROOT=/home/548/${USER}/aus-historical-solar-droughts
export MODULES=${ROOT}/modules

# Append to our python path
export PYTHONPATH=${MODULES}:${PYTHONPATH}