#!/bin/bash 
#PBS -l walltime=1:00:00 
#PBS -l mem=1GB 
#PBS -l ncpus=1
#PBS -l jobfs=10GB 
#PBS -l storage=gdata/xp65+gdata/er8+scratch/er8+gdata/rv74+gdata/rt52+gdata/ob53 
#PBS -l other=hyperthread 
#PBS -q normal 
#PBS -P er8 
#PBS -o /home/548/cd3022/repos/aus-historical-solar-droughts/logs/TEST.oe 
#PBS -j oe
cd /home/548/cd3022/repos/aus-historical-solar-droughts
module use /g/data/xp65/public/modules
module load conda/analysis3

source ./env.sh
echo "PATH BEFORE=$PATH"
source /home/548/cd3022/repos/aus-historical-solar-droughts/pvlib_venv/bin/activate
echo "PATH AFTER =$PATH"
python3 TEST.py