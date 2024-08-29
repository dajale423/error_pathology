#!/bin/bash
#SBATCH -c 2                               # Request one core
#SBATCH -t 0-1:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_requeue                           # Partition to run in gpu_requeue or gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MiB (for all cores)
#SBATCH -o ../slurm_output/hostname_%j.out          # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ../slurm_output/hostname_%j.err          # File to which STDERR will be written, including job ID (%j)
               # You can change the filenames given with -o and -e to any filenames you'd like

module load gcc/9.2.0
module load cuda/11.7

# to monitor gpu usage
# /n/cluster/bin/job_gpu_monitor.sh &

python sensitive_direction.py --layer 6 --direction_type naive_random --subtraction itself

# python sensitive_direction.py --layer 8 --direction_type naive_random_to --length_ranges Short
# python sensitive_direction.py --layer 9 --direction_type naive_random_to --length_ranges Short
# python sensitive_direction.py --layer 10 --direction_type naive_random_to --length_ranges Short
# python sensitive_direction.py --layer 11 --direction_type naive_random_to --length_ranges Short
# python sensitive_direction.py --layer 6 --direction_type naive_random_to --length_ranges Short
# python sensitive_direction.py --layer 7 --direction_type naive_random_to --length_ranges Short

# python sensitive_direction.py --layer 6 --direction_type cov_random
# python sensitive_direction.py --layer 6 --direction_type cov_random_to
# python sensitive_direction.py --layer 6 --direction_type real_direction_to
# python sensitive_direction.py --layer 6 --direction_type naive_random real_direction

