#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=06:00:00
#SBATCH --job-name=gpu_c
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=log/conditional/output_%j.txt
#SBATCH --error=log/conditional/error_%j.txt
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

command="python ./conditional/train.py --s $from --f $to --game $game --instance $instance --experiment ../../../scratch/bazzaz.ma/NegativeExample/models --cuda"
echo $command
$command