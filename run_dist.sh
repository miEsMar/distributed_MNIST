#!/bin/bash
#
# This file has been automatically generated.
# DO NOT modify.
#
#SBATCH --account=bsc85
#SBATCH --qos=gp_debug
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=mnist
#SBATCH --error=err.txt
#SBATCH --output=out.txt
#SBATCH --hint=nomultithread
# SBATCH --constraint=highmem
# SBATCH --threads-per-core=1
# SBATCH --ntasks-per-core=1
# SBATCH --time=00-02:00:00
# SBATCH --sockets-per-node=2
# SBATCH --cpu-freq=3000000-3000000:performance


module purge
ml torchcpu


cmd="mpirun -n 2 --bind-to none --verbose"
# cmd="${cmd} --mca pml ucx"
# cmd="${cmd} -x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=ud,rc,sm"
cmd="${cmd} ./build/dist_mnist "

echo "" >&2
echo $( which mpirun ) >&2
echo ${cmd} >&2

${cmd}

