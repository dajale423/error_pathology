#!/bin/bash


for i in {0..11}
do
    python error_eval.py --layer $i
done

for i in {0..11}
do
    python error_eval.py --layer $i --pos 64
done

for i in {0..11}
do
    python error_eval.py --hook_loc z --layer $i
done