#!/bin/bash

python ./vanilla/generate.py --game slide --instance 1 --epochs 5000
python ./rumi/generate.py --game slide --instance 1 --epochs 5000
python ./conditional/generate.py --game slide --instance 1 --epochs 5000

python ./vanilla/generate.py --game cave --instance 1 --epochs 5000
python ./rumi/generate.py --game cave --instance 1 --epochs 5000
python ./conditional/generate.py --game cave --instance 1 --epochs 5000

python ./vanilla/generate.py --game vertical --instance 1 --epochs 6000
python ./rumi/generate.py --game vertical --instance 1 --epochs 6000
python ./conditional/generate.py --game vertical --instance 1 --epochs 6000

python ./vanilla/generate.py --game crates --instance 1 --epochs 6000
python ./rumi/generate.py --game crates --instance 1 --epochs 6000
python ./conditional/generate.py --game crates --instance 1 --epochs 6000
