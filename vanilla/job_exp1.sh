#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=06:00:00
#SBATCH --job-name=exp1_v
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=log/exp1/vanilla/output_%j.txt
#SBATCH --error=log/exp1/vanilla/error_%j.txt
module load anaconda3/2022.05
module load python/3.8.1
pip install -r requirements.txt
if [ $# -lt 4 ]; then
    echo "Usage: $0 <instance> <from> <to> <game> "
    exit 1
fi
# Assign mandatory arguments to variables
instance=$1
from=$2
to=$3
game=$4

command="python ./vanilla/train_exp1.py --s $from --f $to --game $game --instance $instance --experiment ../../../scratch/bazzaz.ma/NegativeExample/models/exp1 --cuda"
echo $command
$command