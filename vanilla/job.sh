#!/bin/bash
# SLURM directives
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=11:59:00
#SBATCH --job-name=myjob
#SBATCH --output=../log/vanilla/output_%j.txt
#SBATCH --error=../log/vanilla/error_%j.txt
module load anaconda3/2022.05
module load python/3.8.1
pip install -r requirements.txt
if [ $# -lt 4 ]; then
    echo "Usage: $0 <game> <from> <to> <instance>"
    exit 1
fi
# Assign mandatory arguments to variables
game=$1
from=$2
to=$3
instance=$4

command="python train.py --s $from --f $to --game $game --instance $instance --experiment ../../../../scratch/bazzaz.ma/NegativeExample/models --cuda"
echo $command
$command