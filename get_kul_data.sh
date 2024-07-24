#!/bin/bash

# Default values
start=0
end=49
epochs=1

# Parse command-line arguments
while getopts s:t:e: flag
do
    case "${flag}" in
        s) start=${OPTARG};;
        t) end=${OPTARG};;
        e) epochs=${OPTARG};;
    esac
done

echo "==== Run $epochs times starting from $start to $end ======"

for ((i=0; i<$epochs; i++))
do
    for ((j=$start; j<=$end; j++))
    do
        echo "==== Running world $j ===="
        python run_rviz_kul.py --world_idx $j
        echo "==== Done with world $j ===="
    done
done