#!/bin/bash

# All layer all positions experiment
for i in {0..11}
do
    python error_eval.py --layer $i
done

# All layers 1 position experiment
for i in {0..11}
do
    python error_eval.py --layer $i --pos 64
done

# Attn SAEs all layers all positions experiment
for i in {0..11}
do
    python error_eval.py --hook_loc z --layer $i
done

# All layers 1 position experiment
for i in {0..11}
do
    python error_eval.py --layer $i --pos 48 --repeat 10
done

python error_eval.py --layer 6 --pos 48 --repeat 500