#!/bin/bash
#SBATCH -c 2                               # Request one core
#SBATCH -t 0-05:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                           # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MiB (for all cores)
#SBATCH -o slurm_output/hostname_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_output/hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

module load gcc/9.2.0
module load cuda/11.7

# to monitor gpu usage
/n/cluster/bin/job_gpu_monitor.sh &

# All layer all positions experiment
# for i in {0..11}
# do
#     printf "$i\n"
#     python error_eval.py --layer $i
# done

# # All layers 1 position experiment
# for i in {0..11}
# do
#     printf "$i\n"
#     python error_eval.py --layer $i --pos 64
# done

# Attn SAEs all layers all positions experiment
for i in {0..11}
do
    printf "$i\n"
    python error_eval.py --hook_loc z --layer $i
done

# # All layers 1 position experiment
# for i in {0..11}
# do
#     python error_eval.py --layer $i --pos 48 --repeat 10
# done

# python error_eval.py --layer 6 --pos 48 --repeat 10
# python error_eval.py --layer 6 --pos 48 --repeat 500