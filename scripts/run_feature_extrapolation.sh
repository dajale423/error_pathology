#!/bin/bash
#SBATCH -c 2                               # Request one core
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                           # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MiB (for all cores)
#SBATCH -o ../slurm_output/hostname_%j.out          # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ../slurm_output/hostname_%j.err          # File to which STDERR will be written, including job ID (%j)
               # You can change the filenames given with -o and -e to any filenames you'd like

module load gcc/9.2.0
module load cuda/11.7

# to monitor gpu usage
/n/cluster/bin/job_gpu_monitor.sh &

# local
python feature_extrapolation.py --layer 6 --e2e ahxwn90o
python feature_extrapolation.py --layer 6 --e2e 43zmudf4
python feature_extrapolation.py --layer 6 --e2e unji5etq
python feature_extrapolation.py --layer 6 --e2e jup3glm9
python feature_extrapolation.py --layer 6 --e2e h9hrelni
python feature_extrapolation.py --layer 6 --e2e 1jy3m5j0
python feature_extrapolation.py --layer 6 --e2e 4nlqrc2y
python feature_extrapolation.py --layer 6 --e2e 2wvu1zs5
python feature_extrapolation.py --layer 6 --e2e uiwt81f1

