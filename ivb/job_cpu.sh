#!/bin/bash

#PBS -q knsc-q1
#PBS -l select=1:ncpus=20
#PBS -l walltime=1:00:00
#PBS -j oe

date
hostname
cd ${PBS_O_WORKDIR}

date;ulimit -s unlimited;
export OMP_NUM_THREADS=10; export KMP_AFFINITY=granularity=fine,compact;
mpirun -np 1 numactl --localalloc ../bem-bb-SCM.out ../bem-bb_inputs/input_10ts.pbs
date
