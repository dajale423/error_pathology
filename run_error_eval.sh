#!/bin/bash

# Run the error evaluation script iterative over all layers
for i in {0..11}
do
    python error_eval.py --layer $i --pos 64
done