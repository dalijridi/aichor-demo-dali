#!/bin/bash

# Simple loop to create CPU and memory load
for i in {1..999999}; do
    # Create memory load by storing command output in array
    array+=( $(seq 1 1000) )
    
    # Create CPU load with heavy calculations
    for j in {1..1000}; do
        echo "scale=5000; 4*a(1)" | bc -l >/dev/null
    done
    
    # Print progress every 100 iterations
    if [ $((i % 100)) -eq 0 ]; then
        echo "Completed $i iterations"
    fi
done
echo "Process complete"
