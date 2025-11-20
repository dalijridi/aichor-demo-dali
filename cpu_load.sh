#!/bin/bash

# Create a CPU-intensive load using a basic loop
for i in {1..10}; do
    # Calculate squares of numbers to create CPU load
    for j in {1..10}; do
        echo "$j * $j * $j * $j * $j * $j" | bc 
        # > /dev/null
    done
    
    # Print progress every 1000 iterations
    if [ $((i % 10)) -eq 0 ]; then
        echo "Completed $i iterations"
    fi
done

echo "Process complete"
