#!/bin/bash

# exp1 
python ./vanilla/generate_exp1.py --game mario --instance 1 --epochs 500 --solution True
python ./rumi/generate_exp1.py --game mario --instance 1 --epochs 500 --solution True
python ./conditional/generate_exp1.py --game mario --instance 1 --epochs 500 --solution True

python ./vanilla/generate_exp1.py --game cave_treasures --instance 1 --epochs 500 --solution True
python ./rumi/generate_exp1.py --game cave_treasures --instance 1 --epochs 500 --solution True
python ./conditional/generate_exp1.py --game cave_treasures --instance 1 --epochs 500 --solution True

# exp2 
python ./vanilla/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 1
python ./vanilla/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 2
python ./vanilla/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 3

python ./rumi/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 1
python ./rumi/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 2
python ./rumi/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 3

python ./conditional/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 1
python ./conditional/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 2
python ./conditional/generate_exp2.py --game mario --instance 1 --epochs 500 --solution True --cond 3

python ./vanilla/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 1
python ./vanilla/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 2
python ./vanilla/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 3

python ./rumi/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 1
python ./rumi/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 2
python ./rumi/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 3

python ./conditional/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 1
python ./conditional/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 2
python ./conditional/generate_exp2.py --game cave_treasures --instance 1 --epochs 500 --solution True --cond 3


# python ./vanilla/generate_exp1.py --game platform --instance 1 --epochs 5000 --solution True
# python ./rumi/generate_exp1.py --game platform --instance 1 --epochs 5000 --solution True
# python ./conditional/generate_exp1.py --game platform --instance 1 --epochs 5000 --solution True

# python ./vanilla/generate_exp1.py --game slide --instance 1 --epochs 5000 --solution True
# python ./rumi/generate_exp1.py --game slide --instance 1 --epochs 5000 --solution True
# python ./conditional/generate_exp1.py --game slide --instance 1 --epochs 5000 --solution True

# python ./vanilla/generate_exp1.py --game cave --instance 1 --epochs 5000 --solution True
# python ./rumi/generate_exp1.py --game cave --instance 1 --epochs 5000 --solution True
# python ./conditional/generate_exp1.py --game cave --instance 1 --epochs 5000 --solution True

# python ./vanilla/generate_exp1.py --game vertical --instance 1 --epochs 6000 --solution True
# python ./rumi/generate_exp1.py --game vertical --instance 1 --epochs 6000 --solution True
# python ./conditional/generate_exp1.py --game vertical --instance 1 --epochs 6000 --solution True

# python ./vanilla/generate_exp1.py --game crates --instance 1 --epochs 6000
# python ./rumi/generate_exp1.py --game crates --instance 1 --epochs 6000
# python ./conditional/generate_exp1.py --game crates --instance 1 --epochs 6000
