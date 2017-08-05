#!/bin/sh

#PBS -q h-debug
#PBS -l select=1:mpiprocs=1
#PBS -W group_list=gi75
#PBS -l walltime=5:00

cd ${PBS_O_WORKDIR}
module purge
module load intel/16.0.4.258 openmpi-gdr/2.1.1/intel cuda
ulimit -s 1000000
export OMP_NUM_THREADS=1
env
date
echo "======== ======== ======== ======== ======== ======== ======== ========"
mpirun -np 1 ./bem-bb-SCM.out ./input_10ts.pbf 2>&1 | tee log_10ts.txt
echo "======== ======== ======== ======== ======== ======== ======== ========"
date
