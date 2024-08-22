#!/bin/bash

# Initialize the starting value
start=0
end=10

# Loop until the starting value reaches 700
while [ $end -le 701 ]; do
    echo "$start-$end"
    submit -L logs/ -m 4000 -M 4000 -c 4 python production/run_simulation.py --first-tile $start --last-tile $end --num-samples 10000 --num-batches 100 --out-directory /home/NiclasEich/repos/mandelbrot-challenge/results/ 
    start=$((end + 1))
    end=$((start + 99))
done
