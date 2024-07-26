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

python feature_extrapolation.py --layer 6 --e2e $1 --batch_size 1


# local
#python feature_extrapolation.py --layer 6 --e2e ahxwn90o
#python feature_extrapolation.py --layer 6 --e2e 43zmudf4
#python feature_extrapolation.py --layer 6 --e2e unji5etq
#python feature_extrapolation.py --layer 6 --e2e jup3glm9
#python feature_extrapolation.py --layer 6 --e2e h9hrelni
#python feature_extrapolation.py --layer 6 --e2e 1jy3m5j0
#python feature_extrapolation.py --layer 6 --e2e 4nlqrc2y
#python feature_extrapolation.py --layer 6 --e2e 2wvu1zs5
# python feature_extrapolation.py --layer 6 --e2e uiwt81f1 --batch_size 1

# e2e
#python feature_extrapolation.py --layer 6 --e2e 4zcbb4au
#python feature_extrapolation.py --layer 6 --e2e zgdpkafo
#python feature_extrapolation.py --layer 6 --e2e tvj2owza
#python feature_extrapolation.py --layer 6 --e2e 1bubkmps
#python feature_extrapolation.py --layer 6 --e2e wzzcimkj
#python feature_extrapolation.py --layer 6 --e2e 4d5ksz89

# downstream
#python feature_extrapolation.py --layer 6 --e2e 55get0e5
#python feature_extrapolation.py --layer 6 --e2e 2lzle2f0
#python feature_extrapolation.py --layer 6 --e2e zidd0d3y
#python feature_extrapolation.py --layer 6 --e2e rb8q8czb
#python feature_extrapolation.py --layer 6 --e2e p9zmh62k
#python feature_extrapolation.py --layer 6 --e2e 7nkdr21r
#python feature_extrapolation.py --layer 6 --e2e fqdgjxfe
