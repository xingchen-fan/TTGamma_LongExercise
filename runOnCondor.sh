#!/usr/bin/env bash

tar -zxf ttgenv.tar.gz
source ttgenv/bin/activate

tar -zxf ttgamma.tar.gz

python runFullDataset.py $1 --condor --chunksize 100000

