#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=06:00:00
#SBATCH --job-name=exp3_c
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=log/exp3/conditional/output_%j.txt
#SBATCH --error=log/exp3/conditional/error_%j.txt
module load anaconda3/2022.05
module load python/3.8.1
pip install -r requirements.txt
if [ $# -lt 5 ]; then
    echo "Usage: $0 <instance> <from> <to> <game> <cond>"
    exit 1
fi
# Assign mandatory arguments to variables
instance=$1
from=$2
to=$3
game=$4
cond=$5

command="python ./conditional/train_exp3.py --s $from --f $to --game $game --instance $instance --cond $cond --experiment ../../../scratch/bazzaz.ma/NegativeExample/models/exp3 --cuda"
echo $command
$command