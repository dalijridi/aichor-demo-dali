#!/bin/bash

# Generate CPU and memory-intensive load
for i in {1..1000}; do
    # Create a large string to consume memory
    large_string=""
    for j in {1..500}; do
        # Append a random string repeatedly
        large_string+=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 100)
    done
    
    # Perform a CPU-intensive computation (repeated calculations)
    for j in {1..10000}; do
        echo "scale=10; sqrt($j) * $j * log($j)" | bc > /dev/null
    done
    
    # Print progress every 50 iterations
    if [ $((i % 50)) -eq 0 ]; then
        echo "Completed $i iterations"
    fi
done

echo "Process complete"
