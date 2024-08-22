#!/bin/bash

# Initialize the starting value
start=0
end=10

# Loop until the starting value reaches 700
while [ $end -le 700 ]; do
    echo "$start-$end"
    #submit -m 4000 -M 6000 -c 4 python production/run_simulation.py --first-tile $start --last-tile $end --num-samples 100 --num-batches 5 --out-directory /home/NiclasEich/repos/mandelbrot-challenge/results/ \
    start=$((start + 11))
    end=$((end + 11))
done
